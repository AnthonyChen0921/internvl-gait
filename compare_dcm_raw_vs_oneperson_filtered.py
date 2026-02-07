import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NAME_RE = re.compile(r"^HSMR-(DCM_(\d+)_([A-Za-z]*WalkData)_(\d+)_clip)\.npy$")


def parse_name(filename: str) -> Tuple[str, Optional[int], Optional[str], Optional[int]]:
    m = NAME_RE.match(filename)
    if not m:
        seq_id = filename.replace("HSMR-", "").replace(".npy", "")
        return seq_id, None, None, None
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


@dataclass
class Row:
    seq_id: str
    subject_id: Optional[int]
    test_type: Optional[str]
    run: Optional[int]

    raw_path: str
    raw_exists: bool
    raw_frames_T: Optional[int]
    raw_frame_dict_keys: str
    raw_people_min: Optional[int]
    raw_people_p50: Optional[float]
    raw_people_max: Optional[int]
    raw_frames_with_0p: Optional[int]
    raw_frames_with_1p: Optional[int]
    raw_frames_with_2p: Optional[int]
    raw_frames_with_3p_plus: Optional[int]

    filtered_path: str
    filtered_exists: bool
    filtered_T: Optional[int]
    filtered_D: Optional[int]
    filtered_dtype: str

    T_matches: Optional[bool]
    note: str


def _people_count_from_frame_dict(frame_obj: Dict) -> int:
    """
    Raw HSMR frame dict contains 'poses' with shape [P, ...] where P is number of people.
    """
    poses = frame_obj.get("poses", None)
    if poses is None:
        return 0
    try:
        return int(np.asarray(poses).shape[0])
    except Exception:
        return 0


def load_raw_people_counts(raw_path: str) -> Tuple[int, List[int], str]:
    arr = np.load(raw_path, allow_pickle=True)
    # raw is expected: object array length T, each entry is dict
    T = int(arr.shape[0]) if isinstance(arr, np.ndarray) else 0
    keys = ""
    counts: List[int] = []
    if T > 0:
        first = arr[0]
        if isinstance(first, dict):
            keys = ",".join(sorted(list(first.keys())))
        for t in range(T):
            obj = arr[t]
            if isinstance(obj, dict):
                counts.append(_people_count_from_frame_dict(obj))
            else:
                counts.append(0)
    return T, counts, keys


def load_filtered_shape(filtered_path: str) -> Tuple[int, int, str]:
    arr = np.load(filtered_path, allow_pickle=True)
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return 0, 0, str(getattr(arr, "dtype", ""))
    return int(arr.shape[0]), int(arr.shape[1]), str(arr.dtype)


def summarize_counts(counts: List[int]) -> Dict:
    if not counts:
        return {
            "min": None,
            "p50": None,
            "max": None,
            "n0": 0,
            "n1": 0,
            "n2": 0,
            "n3p": 0,
        }
    c = np.asarray(counts, dtype=np.int32)
    return {
        "min": int(c.min()),
        "p50": float(np.quantile(c, 0.5)),
        "max": int(c.max()),
        "n0": int((c == 0).sum()),
        "n1": int((c == 1).sum()),
        "n2": int((c == 2).sum()),
        "n3p": int((c >= 3).sum()),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="GAVD-HSMR-dcm")
    parser.add_argument("--filtered-dir", type=str, default="GAVD-HSMR-dcm-oneperson-filtered")
    parser.add_argument("--out-csv", type=str, default="reports/dcm/dcm_raw_vs_oneperson_filtered.csv")
    parser.add_argument("--out-json", type=str, default="reports/dcm/dcm_raw_vs_oneperson_filtered_summary.json")
    args = parser.parse_args()

    raw_dir = args.raw_dir if os.path.isabs(args.raw_dir) else os.path.join(BASE_DIR, args.raw_dir)
    filtered_dir = (
        args.filtered_dir if os.path.isabs(args.filtered_dir) else os.path.join(BASE_DIR, args.filtered_dir)
    )
    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(BASE_DIR, args.out_csv)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(BASE_DIR, args.out_json)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    raw_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(".npy")])
    filtered_set = set([f for f in os.listdir(filtered_dir) if f.lower().endswith(".npy")])

    rows: List[Row] = []
    for f in raw_files:
        seq_id, subject_id, test_type, run = parse_name(f)
        raw_path = os.path.join(raw_dir, f)
        filtered_path = os.path.join(filtered_dir, f)

        raw_exists = os.path.exists(raw_path)
        filtered_exists = f in filtered_set and os.path.exists(filtered_path)

        note = ""
        raw_T = None
        keys = ""
        ppl_stats = {"min": None, "p50": None, "max": None, "n0": None, "n1": None, "n2": None, "n3p": None}
        if raw_exists:
            try:
                raw_T, counts, keys = load_raw_people_counts(raw_path)
                ppl_stats = summarize_counts(counts)
            except Exception as e:
                note = f"raw_load_error:{type(e).__name__}"

        filt_T = None
        filt_D = None
        filt_dtype = ""
        if filtered_exists:
            try:
                filt_T, filt_D, filt_dtype = load_filtered_shape(filtered_path)
                if filt_D != 46:
                    note = (note + ";" if note else "") + f"filtered_D!=46({filt_D})"
            except Exception as e:
                note = (note + ";" if note else "") + f"filtered_load_error:{type(e).__name__}"

        T_matches = None
        if raw_T is not None and filt_T is not None and raw_T > 0 and filt_T > 0:
            T_matches = bool(raw_T == filt_T)
            if not T_matches:
                note = (note + ";" if note else "") + "T_mismatch"

        rows.append(
            Row(
                seq_id=seq_id,
                subject_id=subject_id,
                test_type=test_type,
                run=run,
                raw_path=raw_path,
                raw_exists=raw_exists,
                raw_frames_T=raw_T,
                raw_frame_dict_keys=keys,
                raw_people_min=ppl_stats["min"],
                raw_people_p50=ppl_stats["p50"],
                raw_people_max=ppl_stats["max"],
                raw_frames_with_0p=ppl_stats["n0"],
                raw_frames_with_1p=ppl_stats["n1"],
                raw_frames_with_2p=ppl_stats["n2"],
                raw_frames_with_3p_plus=ppl_stats["n3p"],
                filtered_path=filtered_path,
                filtered_exists=filtered_exists,
                filtered_T=filt_T,
                filtered_D=filt_D,
                filtered_dtype=filt_dtype,
                T_matches=T_matches,
                note=note,
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(out_csv, index=False)

    # Summary stats
    summary: Dict = {
        "raw_dir": os.path.relpath(raw_dir, BASE_DIR).replace("\\", "/"),
        "filtered_dir": os.path.relpath(filtered_dir, BASE_DIR).replace("\\", "/"),
        "raw_files_total": int(len(raw_files)),
        "filtered_files_total": int(len(filtered_set)),
        "raw_only": int((df["filtered_exists"] == False).sum()),
        "both": int(((df["raw_exists"] == True) & (df["filtered_exists"] == True)).sum()),
    }

    both_df = df[(df["raw_exists"] == True) & (df["filtered_exists"] == True)].copy()
    if len(both_df) > 0:
        summary["T_matches_rate"] = float(both_df["T_matches"].fillna(False).mean())
        # People count distribution in raw
        summary["raw_people_max_value_counts"] = (
            both_df["raw_people_max"].fillna(-1).astype(int).value_counts().sort_index().to_dict()
        )
        summary["raw_any_2p_frames"] = int((both_df["raw_frames_with_2p"].fillna(0).astype(int) > 0).sum())
        summary["raw_any_3p_plus_frames"] = int((both_df["raw_frames_with_3p_plus"].fillna(0).astype(int) > 0).sum())

    # sequences filtered out
    raw_only_df = df[df["filtered_exists"] == False].copy()
    summary["filtered_out_seq_ids_head"] = raw_only_df["seq_id"].head(30).tolist()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Compare raw vs oneperson-filtered ===")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Raw total: {summary['raw_files_total']} | Filtered total: {summary['filtered_files_total']}")
    print(f"Both: {summary['both']} | Raw-only (filtered out): {summary['raw_only']}")
    if "T_matches_rate" in summary:
        print(f"T match rate (both): {summary['T_matches_rate']*100:.2f}%")


if __name__ == "__main__":
    main()


