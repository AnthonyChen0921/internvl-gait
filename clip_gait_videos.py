import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def find_ffmpeg() -> Optional[str]:
    """
    Return a path to ffmpeg if available, else None.
    Tries:
      1) ffmpeg on PATH
      2) imageio_ffmpeg (if installed)
    """
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def format_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


@dataclass
class Mark:
    in_sec: Optional[float] = None
    out_sec: Optional[float] = None


def overlay_text(img: np.ndarray, lines: List[str]) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required for interactive marking. Install with: pip install opencv-python"
        )
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    pad = 8

    x = pad
    y = pad + 20
    for line in lines:
        # shadow
        cv2.putText(out, line, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 22
    return out


def interactive_mark_video(video_path: str, start_paused: bool = True) -> Tuple[Optional[float], Optional[float]]:
    """
    Open an OpenCV window to mark IN/OUT points.

    Controls:
      - SPACE: play/pause
      - i: mark IN at current time
      - o: mark OUT at current time
      - a / d: step -1 / +1 frame (when paused)
      - j / l: jump -1s / +1s
      - k / ;: jump -5s / +5s
      - r: reset marks
      - q or ESC: quit (returns whatever was marked)
    """
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required for interactive marking. Install with: pip install opencv-python"
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = frame_count / fps if frame_count > 0 else 0.0

    win = "Clipper (SPACE play/pause | i IN | o OUT | q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    playing = not start_paused
    mark = Mark()

    def get_pos_sec() -> float:
        fi = cap.get(cv2.CAP_PROP_POS_FRAMES)
        return float(fi) / float(fps)

    def set_pos_sec(sec: float):
        sec = float(np.clip(sec, 0.0, max(0.0, duration - 1e-6) if duration > 0 else 0.0))
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)

    # Prime first frame
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read frames from: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        if playing:
            ok, frame = cap.read()
            if not ok:
                playing = False
                # stay at end
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
                continue
        else:
            # read current frame without advancing too much: set -> read -> set back
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok, frame = cap.read()
            if not ok:
                # sometimes reading at end fails; try one frame back
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 1))
                ok, frame = cap.read()
                if not ok:
                    break
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)

        pos = get_pos_sec()
        lines = [
            os.path.basename(video_path),
            f"t={format_time(pos)} / {format_time(duration)}  (fps={fps:.2f}, frames={frame_count})",
            f"IN:  {format_time(mark.in_sec) if mark.in_sec is not None else '--'}",
            f"OUT: {format_time(mark.out_sec) if mark.out_sec is not None else '--'}",
            "SPACE play/pause | i IN | o OUT | a/d frame | j/l 1s | k/; 5s | r reset | q quit",
        ]
        vis = overlay_text(frame, lines)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1 if playing else 0) & 0xFF

        if key in (27, ord("q")):  # ESC or q
            break
        if key == ord(" "):
            playing = not playing
            continue
        if key == ord("i"):
            mark.in_sec = pos
            continue
        if key == ord("o"):
            mark.out_sec = pos
            continue
        if key == ord("r"):
            mark = Mark()
            continue

        # seek controls
        if key == ord("a") and not playing:
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 1))
            continue
        if key == ord("d") and not playing:
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, frame_count - 1), cur + 1))
            continue
        if key == ord("j"):
            set_pos_sec(pos - 1.0)
            continue
        if key == ord("l"):
            set_pos_sec(pos + 1.0)
            continue
        if key == ord("k"):
            set_pos_sec(pos - 5.0)
            continue
        if key == ord(";"):
            set_pos_sec(pos + 5.0)
            continue

    cap.release()
    cv2.destroyWindow(win)

    # normalize in/out ordering if both exist
    if mark.in_sec is not None and mark.out_sec is not None and mark.out_sec < mark.in_sec:
        mark.in_sec, mark.out_sec = mark.out_sec, mark.in_sec
    return mark.in_sec, mark.out_sec


def ffmpeg_trim(
    ffmpeg: str,
    in_path: str,
    out_path: str,
    start_sec: float,
    end_sec: float,
    reencode: bool,
    audio: bool,
):
    """
    Trim using ffmpeg.
    - If reencode=False, uses stream copy (fast but cut points can be less exact).
    - If reencode=True, re-encodes for accurate cut points.
    """
    safe_mkdir(os.path.dirname(out_path) or ".")

    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))

    # Use input seeking for speed; for accurate cuts, reencode.
    cmd = [ffmpeg, "-hide_banner", "-y", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}", "-i", in_path]
    if reencode:
        # Use H.264 + AAC; good default compatibility.
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
    else:
        cmd += ["-c", "copy"]
        if not audio:
            cmd += ["-an"]

    cmd += [out_path]
    subprocess.run(cmd, check=True)


def opencv_trim(
    in_path: str,
    out_path: str,
    start_sec: float,
    end_sec: float,
):
    """
    Fallback trim without ffmpeg (re-encodes via OpenCV; slower).
    Writes .mp4 with mp4v.
    """
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required for the OpenCV fallback trimming. Install with: pip install opencv-python"
        )
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    safe_mkdir(os.path.dirname(out_path) or ".")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for: {out_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)
    while True:
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec >= end_sec * 1000.0:
            break
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    writer.release()
    cap.release()


def list_videos_in_dir(root: str) -> List[str]:
    out = []
    for fn in os.listdir(root):
        p = os.path.join(root, fn)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in VIDEO_EXTS:
            out.append(p)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively mark and trim gait videos to remove preparation phases."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Single video path to mark+trim. If omitted, you can use --dir to pick by index.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gait-videos"),
        help="Directory containing videos (default: ./Gait-videos).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="If --video not provided, pick the Nth video from --dir (0-based).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Gait-videos-clipped",
        help="Output directory (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=-1.0,
        help="Optional: start time in seconds. If negative, set interactively.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=-1.0,
        help="Optional: end time in seconds. If negative, set interactively.",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode with ffmpeg for accurate cuts (slower, larger compute).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Drop audio track (if using ffmpeg).",
    )
    parser.add_argument(
        "--save-marks-csv",
        type=str,
        default="",
        help="Optional: append marks to a CSV with columns: path,start_sec,end_sec,out_path",
    )
    parser.add_argument(
        "--apply-csv",
        type=str,
        default="",
        help="Batch mode: apply trims from a CSV (columns: path,start_sec,end_sec[,out_path]).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo_root, args.out_dir)

    ffmpeg = find_ffmpeg()
    if cv2 is None and not ffmpeg:
        raise ModuleNotFoundError(
            "Neither OpenCV (cv2) nor ffmpeg was found. Install at least one:\n"
            "  - pip install opencv-python\n"
            "  - or install ffmpeg (or pip install imageio-ffmpeg)\n"
        )

    if args.apply_csv:
        csv_path = args.apply_csv if os.path.isabs(args.apply_csv) else os.path.join(repo_root, args.apply_csv)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"Batch applying trims from: {csv_path}")
        if ffmpeg:
            print(f"Using ffmpeg: {ffmpeg}")
        else:
            print("ffmpeg not found; falling back to OpenCV re-encode (slower).")

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                in_path = row["path"]
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
                out_path = row.get("out_path", "").strip()
                if not out_path:
                    base = os.path.splitext(os.path.basename(in_path))[0]
                    out_path = os.path.join(out_dir, f"{base}_clip.mp4")
                if not os.path.isabs(out_path):
                    out_path = os.path.join(repo_root, out_path)

                print(f"- {os.path.basename(in_path)}: {start_sec:.2f}s -> {end_sec:.2f}s  =>  {out_path}")
                if ffmpeg:
                    ffmpeg_trim(
                        ffmpeg=ffmpeg,
                        in_path=in_path,
                        out_path=out_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        reencode=args.reencode,
                        audio=not args.no_audio,
                    )
                else:
                    opencv_trim(in_path=in_path, out_path=out_path, start_sec=start_sec, end_sec=end_sec)
        return

    # Determine which video to open
    video_path = args.video
    if not video_path:
        d = args.dir if os.path.isabs(args.dir) else os.path.join(repo_root, args.dir)
        vids = list_videos_in_dir(d)
        if not vids:
            raise FileNotFoundError(f"No videos found in: {d}")
        if args.index < 0:
            print("Pick a video by index:")
            for i, p in enumerate(vids[:50]):
                print(f"  [{i}] {os.path.basename(p)}")
            if len(vids) > 50:
                print(f"  ... and {len(vids) - 50} more")
            raise SystemExit("Re-run with --index N or --video PATH")
        if args.index >= len(vids):
            raise IndexError(f"--index {args.index} out of range (0..{len(vids)-1})")
        video_path = vids[args.index]

    if not os.path.isabs(video_path):
        video_path = os.path.join(repo_root, video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Mark IN/OUT times
    start_sec = args.start if args.start >= 0 else None
    end_sec = args.end if args.end >= 0 else None

    if start_sec is None or end_sec is None:
        in_sec, out_sec = interactive_mark_video(video_path, start_paused=True)
        if start_sec is None:
            start_sec = in_sec
        if end_sec is None:
            end_sec = out_sec

    if start_sec is None or end_sec is None:
        raise SystemExit("You must provide both start and end. Use keys i/o to mark IN/OUT, then press q to quit.")
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec

    # Output path
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_clip.mp4")

    print("\n=== Trim ===")
    print(f"Input:  {video_path}")
    print(f"Start:  {start_sec:.3f}s")
    print(f"End:    {end_sec:.3f}s")
    print(f"Output: {out_path}")
    if ffmpeg:
        print(f"Using ffmpeg: {ffmpeg}  (reencode={args.reencode}, audio={not args.no_audio})")
        ffmpeg_trim(
            ffmpeg=ffmpeg,
            in_path=video_path,
            out_path=out_path,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            reencode=args.reencode,
            audio=not args.no_audio,
        )
    else:
        print("ffmpeg not found; using OpenCV fallback (slower, no audio).")
        opencv_trim(in_path=video_path, out_path=out_path, start_sec=float(start_sec), end_sec=float(end_sec))

    # Optionally append to marks CSV
    if args.save_marks_csv:
        csv_path = args.save_marks_csv if os.path.isabs(args.save_marks_csv) else os.path.join(repo_root, args.save_marks_csv)
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "start_sec", "end_sec", "out_path"])
            if not file_exists:
                w.writeheader()
            w.writerow({"path": video_path, "start_sec": start_sec, "end_sec": end_sec, "out_path": out_path})
        print(f"Appended marks to: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


