import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}


@dataclass(frozen=True)
class ParsedName:
    cohort: str
    subject_id: str
    test_type: str
    run: str


_PATTERNS: Sequence[re.Pattern] = [
    # DCM_19_WalkData_1 / DCM_19_FastWalkData_2 / DCM_55_FwdWalkData_1 / DCM_68_Test_1
    re.compile(
        r"^(?P<cohort>[A-Za-z]+)_(?P<subject>\d{1,6})_(?P<type>[A-Za-z]+(?:Walk)?Data|[A-Za-z]+)_(?P<run>\d+)$",
        re.IGNORECASE,
    ),
    # Same as above but non-numeric run suffix: e.g. WalkData_Fast
    re.compile(
        r"^(?P<cohort>[A-Za-z]+)_(?P<subject>\d{1,6})_(?P<type>[A-Za-z]+(?:Walk)?Data|[A-Za-z]+)_(?P<run>[A-Za-z]+)$",
        re.IGNORECASE,
    ),
    # 21_T5_WalkData_Fast  -> cohort=T5, subject=21, type=WalkData, run=Fast
    re.compile(
        r"^(?P<subject>\d{1,6})_T(?P<t>\d{1,3})_(?P<type>[A-Za-z]+Data|[A-Za-z]+)_(?P<run>[A-Za-z0-9]+)$",
        re.IGNORECASE,
    ),
]


def parse_stem(stem: str) -> Optional[ParsedName]:
    stem = stem.strip()
    for pat in _PATTERNS:
        m = pat.match(stem)
        if not m:
            continue
        gd = m.groupdict()
        if "t" in gd:
            cohort = f"T{gd['t']}".upper()
            subject_id = str(gd["subject"])
            test_type = str(gd["type"])
            run = str(gd["run"])
            return ParsedName(cohort=cohort, subject_id=subject_id, test_type=test_type, run=run)

        cohort = str(gd.get("cohort", "")).upper()
        subject_id = str(gd["subject"])
        test_type = str(gd["type"])
        run = str(gd["run"])
        return ParsedName(cohort=cohort, subject_id=subject_id, test_type=test_type, run=run)
    return None


def iter_video_files(root: str, recursive: bool, exclude_dirs: Iterable[str]) -> Iterable[str]:
    exclude_dirs_norm = {d.replace("/", "\\").rstrip("\\").lower() for d in exclude_dirs}
    root = os.path.abspath(root)
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirpath_norm = os.path.abspath(dirpath).lower()
            # prune excluded dirs
            pruned = []
            for d in list(dirnames):
                full = os.path.abspath(os.path.join(dirpath, d)).lower()
                if any(full.startswith(ex) for ex in exclude_dirs_norm):
                    pruned.append(d)
            for d in pruned:
                dirnames.remove(d)

            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in VIDEO_EXTS:
                    yield os.path.join(dirpath, fn)
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in VIDEO_EXTS:
                yield p


def try_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute subject/type/run statistics from gait video filenames (e.g., DCM_<id>_<Type>_<run>.MOV)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gait-videos"),
        help="Directory containing gait videos (default: ./Gait-videos).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories.")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory prefix to exclude (can be specified multiple times).",
    )
    parser.add_argument("--out-json", type=str, default="", help="Optional: write JSON summary to this path.")
    parser.add_argument("--out-csv", type=str, default="", help="Optional: write parsed file list to this CSV path.")
    parser.add_argument(
        "--show-unparsed",
        type=int,
        default=25,
        help="How many unparsed filenames to print as examples (default: 25).",
    )
    args = parser.parse_args()

    root = args.dir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    # Always exclude the common zip folder unless user overrides
    exclude_dirs = list(args.exclude_dir)
    zip_dir = os.path.abspath(os.path.join(root, "zip"))
    if zip_dir.lower() not in {os.path.abspath(d).lower() for d in exclude_dirs}:
        exclude_dirs.append(zip_dir)

    files = list(iter_video_files(root, recursive=args.recursive, exclude_dirs=exclude_dirs))
    parsed_rows: List[Dict[str, str]] = []
    unparsed: List[str] = []

    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        parsed = parse_stem(stem)
        if parsed is None:
            unparsed.append(os.path.basename(p))
            continue
        parsed_rows.append(
            {
                "path": p,
                "cohort": parsed.cohort,
                "subject_id": parsed.subject_id,
                "type": parsed.test_type,
                "run": parsed.run,
            }
        )

    # Stats
    subjects: Set[Tuple[str, str]] = set((r["cohort"], r["subject_id"]) for r in parsed_rows)
    types: Set[str] = set(r["type"] for r in parsed_rows)

    type_to_subjects: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    type_to_runs: Dict[str, Set[str]] = defaultdict(set)
    type_to_files: Counter = Counter()

    # (type, cohort, subject) -> runs set
    runs_by_subject_type: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

    for r in parsed_rows:
        key_subj = (r["cohort"], r["subject_id"])
        t = r["type"]
        type_to_subjects[t].add(key_subj)
        type_to_runs[t].add(r["run"])
        type_to_files[t] += 1
        runs_by_subject_type[(t, r["cohort"], r["subject_id"])].add(r["run"])

    # Missing-run detection for numeric runs (per subject+type)
    missing_runs_by_subject_type: Dict[Tuple[str, str, str], List[int]] = {}
    for (t, cohort, subject), runs in runs_by_subject_type.items():
        nums = sorted([x for x in (try_int(r) for r in runs) if x is not None])
        if len(nums) < 2:
            continue
        exp = set(range(min(nums), max(nums) + 1))
        miss = sorted(exp - set(nums))
        if miss:
            missing_runs_by_subject_type[(t, cohort, subject)] = miss

    # Print report
    print("\n=== Gait video naming stats ===")
    print(f"Root: {os.path.abspath(root)}")
    print(f"Files scanned (video extensions): {len(files)}")
    print(f"Parsed files: {len(parsed_rows)}")
    print(f"Unparsed files: {len(unparsed)}")

    print(f"\nUnique subjects (cohort+id): {len(subjects)}")
    print(f"Unique test types: {len(types)}")

    print("\nPer-type summary (files, subjects, unique runs):")
    for t in sorted(types):
        print(
            f"  {t}: files={type_to_files[t]}, "
            f"subjects={len(type_to_subjects[t])}, "
            f"runs={len(type_to_runs[t])}"
        )

    # Runs per type (sorted)
    print("\nRuns observed per type:")
    for t in sorted(types):
        runs = sorted(type_to_runs[t], key=lambda x: (try_int(x) is None, try_int(x) if try_int(x) is not None else x))
        # shorten display if huge
        if len(runs) > 30:
            head = ", ".join(runs[:15])
            tail = ", ".join(runs[-10:])
            print(f"  {t}: [{head}, ... , {tail}] (total {len(runs)})")
        else:
            print(f"  {t}: {runs}")

    if missing_runs_by_subject_type:
        print("\nDetected missing numeric runs (per subject+type) examples:")
        shown = 0
        for (t, cohort, subject), miss in sorted(missing_runs_by_subject_type.items())[:25]:
            print(f"  {cohort}_{subject} {t}: missing {miss}")
            shown += 1
        if len(missing_runs_by_subject_type) > shown:
            print(f"  ... and {len(missing_runs_by_subject_type) - shown} more")

    if unparsed and args.show_unparsed > 0:
        print(f"\nUnparsed filename examples (first {min(args.show_unparsed, len(unparsed))}):")
        for fn in unparsed[: args.show_unparsed]:
            print(f"  {fn}")

    # Write outputs
    summary = {
        "root": os.path.abspath(root),
        "files_scanned": len(files),
        "parsed_files": len(parsed_rows),
        "unparsed_files": len(unparsed),
        "unique_subjects": len(subjects),
        "unique_types": len(types),
        "types": sorted(types),
        "per_type": {
            t: {
                "files": int(type_to_files[t]),
                "subjects": int(len(type_to_subjects[t])),
                "unique_runs": int(len(type_to_runs[t])),
                "runs": sorted(list(type_to_runs[t])),
            }
            for t in sorted(types)
        },
        "missing_numeric_runs_count": len(missing_runs_by_subject_type),
    }

    if args.out_json:
        out_json = os.path.abspath(args.out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote JSON summary to: {out_json}")

    if args.out_csv:
        out_csv = os.path.abspath(args.out_csv)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "cohort", "subject_id", "type", "run"])
            writer.writeheader()
            writer.writerows(parsed_rows)
        print(f"Wrote parsed file list CSV to: {out_csv}")


if __name__ == "__main__":
    main()




