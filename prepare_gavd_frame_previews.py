import os
import glob

import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")
PREVIEW_DIR = os.path.join(BASE_DIR, "GAVD-frames-preview")

# How many videos and frames per video to preview
MAX_VIDEOS = 20
FRAMES_PER_VIDEO = 8


def extract_preview_frames():
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"Found {len(video_paths)} videos in {VIDEO_DIR}")

    for idx, vid_path in enumerate(video_paths[:MAX_VIDEOS]):
        seq_id = os.path.splitext(os.path.basename(vid_path))[0]
        print(f"[{idx+1}/{min(len(video_paths), MAX_VIDEOS)}] Processing {seq_id}.mp4")

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  Warning: could not open {vid_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            print(f"  Warning: no frames detected in {vid_path}")
            cap.release()
            continue

        # Choose FRAMES_PER_VIDEO evenly spaced frame indices
        indices = []
        if frame_count <= FRAMES_PER_VIDEO:
            indices = list(range(frame_count))
        else:
            step = frame_count / FRAMES_PER_VIDEO
            indices = [int(i * step) for i in range(FRAMES_PER_VIDEO)]

        for j, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                print(f"  Warning: could not read frame {frame_idx} in {vid_path}")
                continue

            out_name = f"{seq_id}_f{frame_idx:05d}.jpg"
            out_path = os.path.join(PREVIEW_DIR, out_name)
            # OpenCV uses BGR; it's fine for preview saving.
            cv2.imwrite(out_path, frame)

        cap.release()

    print(f"Preview frames saved to: {PREVIEW_DIR}")


if __name__ == "__main__":
    extract_preview_frames()

































