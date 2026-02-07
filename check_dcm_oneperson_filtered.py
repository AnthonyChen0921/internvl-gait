import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Row:
    seq_id: str
    filename: str
    subject_id: Optional[int]
    test_type: Optional[str]
    run: Optional[int]
    path: str
    exists: bool
    ok: bool
    ndim: Optional[int]
    shape0_T: Optional[int]
    shape1_D: Optional[int]
    dtype: str
    nan_count: Optional[int]
    inf_count: Optional[int]
    min_val: Optional[float]
    max_val: Optional[float]
    note: str


# Note: `WalkData` itself must match (so we allow 0+ prefix letters before "WalkData").
NAME_RE = re.compile(r"^HSMR-(DCM_(\d+)_([A-Za-z]*WalkData)_(\d+)_clip)\.npy$")


def parse_name(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    m = NAME_RE.match(filename)
    if not m:
        return None, None, None, None
    seq_id = m.group(1)
    subject_id = int(m.group(2))
    test_type = m.group(3)
    run = int(m.group(4))
    return seq_id, subject_id, test_type, run


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def check_dir(in_dir: str) -> Tuple[pd.DataFrame, Dict]:
    in_dir = os.path.normpath(in_dir)
    files = [f for f in os.listdir(in_dir) if f.lower().endswith(".npy")]
    files.sort()

    rows: List[Row] = []
    for f in files:
        path = os.path.join(in_dir, f)
        exists = os.path.exists(path)
        seq_id, subject_id, test_type, run = parse_name(f)
        note = ""

        if seq_id is None:
            seq_id = f.replace("HSMR-", "").replace(".npy", "")
            note = "filename_unexpected_format"

        if not exists:
            rows.append(
                Row(
                    seq_id=seq_id,
                    filename=f,
                    subject_id=subject_id,
                    test_type=test_type,
                    run=run,
                    path=path,
                    exists=False,
                    ok=False,
                    ndim=None,
                    shape0_T=None,
                    shape1_D=None,
                    dtype="",
                    nan_count=None,
                    inf_count=None,
                    min_val=None,
                    max_val=None,
                    note=note + (";missing_file" if note else "missing_file"),
                )
            )
            continue

        ok = True
        ndim = None
        T = None
        D = None
        dtype = ""
        nan_count = None
        inf_count = None
        min_val = None
        max_val = None

        try:
            arr = np.load(path, allow_pickle=True)
            ndim = int(arr.ndim)
            dtype = str(arr.dtype)
            if arr.ndim != 2:
                ok = False
                note = note + (";ndim!=2" if note else "ndim!=2")
            else:
                T = int(arr.shape[0])
                D = int(arr.shape[1])
                if D != 46:
                    ok = False
                    note = note + (";D!=46" if note else "D!=46")
                if T <= 0:
                    ok = False
                    note = note + (";T<=0" if note else "T<=0")

            # Numeric sanity checks (only for numeric dtypes)
            if arr.size > 0 and arr.dtype != object:
                nan_count = int(np.isnan(arr).sum())
                inf_count = int(np.isinf(arr).sum())
                if nan_count > 0:
                    ok = False
                    note = note + (";has_nan" if note else "has_nan")
                if inf_count > 0:
                    ok = False
                    note = note + (";has_inf" if note else "has_inf")
                min_val = _safe_float(np.nanmin(arr))
                max_val = _safe_float(np.nanmax(arr))
            else:
                if arr.dtype == object:
                    ok = False
                    note = note + (";dtype_object" if note else "dtype_object")
        except Exception as e:
            ok = False
            note = note + (f";load_error:{type(e).__name__}" if note else f"load_error:{type(e).__name__}")

        rows.append(
            Row(
                seq_id=seq_id,
                filename=f,
                subject_id=subject_id,
                test_type=test_type,
                run=run,
                path=path,
                exists=True,
                ok=ok,
                ndim=ndim,
                shape0_T=T,
                shape1_D=D,
                dtype=dtype,
                nan_count=nan_count,
                inf_count=inf_count,
                min_val=min_val,
                max_val=max_val,
                note=note,
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows])

    # Aggregate stats
    ok_df = df[df["ok"] == True].copy()
    subjects = sorted({int(x) for x in ok_df["subject_id"].dropna().astype(int).tolist()})
    types = sorted({str(x) for x in ok_df["test_type"].dropna().astype(str).tolist()})
    n_subjects = len(subjects)
    n_files = int(df.shape[0])
    n_ok = int(ok_df.shape[0])

    summary: Dict = {
        "in_dir": os.path.relpath(in_dir, BASE_DIR).replace("\\", "/"),
        "files_total": n_files,
        "files_ok": n_ok,
        "files_bad": int(n_files - n_ok),
        "unique_subjects_ok": n_subjects,
        "subjects_ok": subjects,
        "types_ok": types,
    }
    if n_ok > 0:
        Ts = ok_df["shape0_T"].astype(int)
        summary["frames_T_stats_ok"] = {
            "min": int(Ts.min()),
            "p25": float(Ts.quantile(0.25)),
            "median": float(Ts.quantile(0.5)),
            "p75": float(Ts.quantile(0.75)),
            "p95": float(Ts.quantile(0.95)),
            "max": int(Ts.max()),
            "mean": float(Ts.mean()),
        }
        summary["per_type_counts_ok"] = (
            ok_df.groupby("test_type")["seq_id"].count().sort_values(ascending=False).to_dict()
        )
        per_sub = (
            ok_df.dropna(subset=["subject_id"])
            .assign(subject_id=lambda x: x["subject_id"].astype(int))
            .groupby("subject_id")["seq_id"]
            .count()
            .sort_values(ascending=False)
        )
        summary["per_subject_counts_ok_top20"] = per_sub.head(20).to_dict()

    bad_examples = df[df["ok"] == False][["filename", "note"]].head(25).to_dict(orient="records")
    summary["bad_examples_head"] = bad_examples
    return df, summary


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        default="GAVD-HSMR-dcm-oneperson-filtered",
        help="Folder containing HSMR-DCM_XX_*_clip.npy files.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="reports/dcm/dcm_oneperson_filtered_file_check.csv",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="reports/dcm/dcm_oneperson_filtered_file_check_summary.json",
    )
    args = parser.parse_args()

    in_dir = args.in_dir if os.path.isabs(args.in_dir) else os.path.join(BASE_DIR, args.in_dir)
    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(BASE_DIR, args.out_csv)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(BASE_DIR, args.out_json)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df, summary = check_dir(in_dir)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DCM oneperson-filtered check ===")
    print(f"In dir: {in_dir}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Files: total={summary['files_total']} ok={summary['files_ok']} bad={summary['files_bad']}")
    print(f"Unique subjects (ok): {summary['unique_subjects_ok']}")
    if summary.get("files_bad", 0) > 0:
        print("Bad examples (head):")
        for e in summary.get("bad_examples_head", []):
            print(f"  {e['filename']}: {e['note']}")


if __name__ == "__main__":
    main()


