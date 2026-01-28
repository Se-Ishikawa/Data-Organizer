import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterable

import pandas as pd


# =========================
# Defaults
# =========================
DEFAULT_ENCODINGS = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin-1"]
DEFAULT_SEPARATORS = [",", "\t", ";", r"\s*\|\s*", "|"]  # enhanced
DEFAULT_EXT = {".csv", ".txt", ".tsv"}
DEFAULT_MAX_PREVIEW_LINES = 80


# =========================
# Helpers
# =========================
def ensure_dirs(out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = out_dir / "cleaned_csv"
    clean_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    return clean_dir, meta_dir


def iter_target_files(root: Path, exts: Iterable[str], recursive: bool) -> List[Path]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    it = root.rglob("*") if recursive else root.glob("*")
    files: List[Path] = []
    for p in it:
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def _count_separators_in_lines(lines: List[str], seps: List[str], max_lines: int = 30) -> Dict[str, int]:
    counts = {s: 0 for s in seps}
    for line in lines[:max_lines]:
        for s in seps:
            try:
                counts[s] += line.count(s)
            except Exception:
                pass
    return counts


def _split_by_sep(line: str, sep: str) -> List[str]:
    # allow regex separators
    if any(ch in sep for ch in ["\\", "(", "[", "?", "*", "+", "{"]):
        parts = re.split(sep, line)
    else:
        parts = line.split(sep)
    return [p.strip() for p in parts]


# =========================
# Robust header detection (sample3 type)
# =========================
def detect_skiprows_and_header(raw_lines: List[str], seps_to_try: List[str], max_preview_lines: int) -> Tuple[int, Optional[int], Optional[str]]:
    """
    Robust header detection for:
      metadata block (may contain separators) + blank lines + actual table header + data rows

    Returns:
      skiprows: index of detected header row
      header: 0
      chosen_sep: best separator for header detection (informational)
    """
    # NOTE: ';' is NOT treated as comment because sample3 uses ';' in metadata lines
    comment_pat = re.compile(r"^\s*(#|//|---)\s*")

    def is_junk_line(s: str) -> bool:
        if not s.strip():
            return True
        if comment_pat.match(s.strip()):
            return True
        return False

    def non_empty_count(parts: List[str]) -> int:
        return sum(1 for p in parts if p != "")

    def looks_like_data_row(parts: List[str]) -> Tuple[bool, float]:
        vals = [p for p in parts if p != ""]
        if len(vals) < 2:
            return False, 0.0

        num_hit = 0
        for v in vals:
            vv = v.replace(",", "").replace("%", "").strip()
            try:
                float(vv)
                num_hit += 1
            except Exception:
                pass

        ratio = num_hit / max(len(vals), 1)
        return (ratio >= 0.3), ratio

    best = None  # (score, header_idx, sep_used)
    lines = raw_lines[:max_preview_lines]

    for i, line in enumerate(lines):
        s = line.strip()
        if is_junk_line(s):
            continue

        for sep in seps_to_try:
            # quick filter for literal seps
            if not any(ch in sep for ch in ["\\", "(", "[", "?", "*", "+", "{"]):
                if sep not in s:
                    continue

            parts = _split_by_sep(s, sep)
            n_fields = len(parts)
            n_non_empty = non_empty_count(parts)

            # header should have >= 3 meaningful fields
            if n_fields < 3 or n_non_empty < 3:
                continue

            emptiness = 1.0 - (n_non_empty / max(n_fields, 1))
            # reject mostly-empty lines like ';;;;;' or 'Date;2025/01/12;;;;'
            if emptiness > 0.6:
                continue

            # check next few lines for data-likeness
            data_hits = 0
            numeric_ratios = []
            field_consistency = 0

            for j in range(i + 1, min(i + 8, len(lines))):
                sj = lines[j].strip()
                if is_junk_line(sj):
                    continue
                pj = _split_by_sep(sj, sep)
                nn = non_empty_count(pj)
                if nn < 2:
                    continue

                if abs(len(pj) - n_fields) <= 1:
                    field_consistency += 1

                ok, r = looks_like_data_row(pj)
                if ok:
                    data_hits += 1
                    numeric_ratios.append(r)

            if data_hits < 2:
                continue

            avg_numeric = sum(numeric_ratios) / max(len(numeric_ratios), 1)
            score = (n_non_empty * 2.0) + (field_consistency * 1.5) + (data_hits * 2.0) + (avg_numeric * 2.0) - (emptiness * 3.0)

            if best is None or score > best[0]:
                best = (score, i, sep)

    if best is not None:
        _score, header_idx, sep_used = best
        return header_idx, 0, sep_used

    # fallback (original-like)
    skip = 0
    for line in lines:
        s = line.strip()
        if not s:
            skip += 1
            continue
        if comment_pat.match(s):
            skip += 1
            continue
        has_sep_signal = any(sep in s for sep in seps_to_try if not any(ch in sep for ch in ["\\", "(", "[", "?", "*", "+", "{"]))
        if (not has_sep_signal) and (len(s) < 120):
            skip += 1
            continue
        break

    # ---- SAFETY: prevent skiprows from consuming whole file ----
    # If skiprows reaches the end, pandas will raise "No columns to parse from file".
    if skip >= len(raw_lines) - 1:
        # Fall back to no-skip rather than failing hard.
        return 0, 0, None

    return skip, 0, None


# =========================
# Metadata extraction
# =========================
def extract_metadata(raw_lines: List[str], header_idx: int, seps_to_try: List[str]) -> List[Dict[str, str]]:
    """
    Extract key/value metadata from lines BEFORE the detected header line.
    Handles common patterns:
      - "Key: Value"
      - "Key;Value;;;;" (semicolon metadata row)
      - "Key,Value" etc. (if 2-field and rest empty)

    Returns list of dict rows: {"line_no":..., "key":..., "value":..., "raw":...}
    """
    rows: List[Dict[str, str]] = []
    comment_pat = re.compile(r"^\s*(#|//|---)\s*")

    def is_noise(s: str) -> bool:
        if not s.strip():
            return True
        if comment_pat.match(s.strip()):
            return True
        if re.fullmatch(r"[-=]{3,}", s.strip()):
            return True
        return False

    for ln, line in enumerate(raw_lines[:max(header_idx, 0)]):
        s = line.strip()
        if is_noise(s):
            continue

        # pattern 1: "Key: Value"
        if ":" in s:
            k, v = s.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                rows.append({"line_no": str(ln), "key": k, "value": v, "raw": s})
                continue

        # pattern 2: delimited "Key;Value;;;;" etc.
        best = None  # (score, sep, parts)
        for sep in seps_to_try:
            if not any(ch in sep for ch in ["\\", "(", "[", "?", "*", "+", "{"]):
                if sep not in s:
                    continue
            parts = _split_by_sep(s, sep)
            n_non_empty = sum(1 for p in parts if p)
            if n_non_empty < 2:
                continue

            emptiness = 1.0 - (n_non_empty / max(len(parts), 1))
            score = 0.0
            if n_non_empty == 2:
                score += 3.0
            score += emptiness
            if n_non_empty >= 4:
                score -= 2.0

            if best is None or score > best[0]:
                best = (score, sep, parts)

        if best is not None:
            _score, _sep, parts = best
            non_empty = [p for p in parts if p]
            if len(non_empty) >= 2:
                k = non_empty[0]
                v = non_empty[1]
                if k and v and len(k) <= 60:
                    rows.append({"line_no": str(ln), "key": k, "value": v, "raw": s})
                    continue

        # pattern 3: free text note
        if len(s) <= 120:
            rows.append({"line_no": str(ln), "key": "NOTE", "value": s, "raw": s})

    return rows


# =========================
# Cleaning utilities
# =========================
def normalize_columns(cols: List[str]) -> List[str]:
    out: List[str] = []
    seen: Dict[str, int] = {}
    for c in cols:
        if c is None:
            c = "col"
        c2 = str(c).strip()
        c2 = c2.replace("\ufeff", "")
        c2 = re.sub(r"\s+", "_", c2)
        if c2 == "" or c2.lower() in {"nan", "none"}:
            c2 = "col"

        if c2 in seen:
            seen[c2] += 1
            c2 = f"{c2}_{seen[c2]}"
        else:
            seen[c2] = 0

        out.append(c2)
    return out


def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    return df


def try_parse_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    parsed_cols: List[str] = []
    name_candidates: List[str] = []
    value_candidates: List[str] = []

    date_like = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}(\s+\d{1,2}:\d{2}(:\d{2})?)?\s*$")

    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["time", "date", "日時", "timestamp", "時刻"]):
            name_candidates.append(c)

    for c in df.columns:
        if df[c].dtype != "object":
            continue
        s = df[c].dropna().astype(str).head(50).tolist()
        if not s:
            continue
        hit = sum(1 for v in s if date_like.match(v))
        if hit / max(len(s), 1) >= 0.3:
            value_candidates.append(c)

    candidates = list(dict.fromkeys(name_candidates + value_candidates))
    for c in candidates:
        try:
            if df[c].dtype == "object":
                dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                ratio = float(dt.notna().mean()) if len(dt) else 0.0
                if ratio >= 0.2:
                    df[c] = dt
                    parsed_cols.append(str(c))
        except Exception:
            pass

    return df, parsed_cols


def coerce_numeric_columns(df: pd.DataFrame, protect_datetime: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    converted: List[str] = []
    for c in df.columns:
        if protect_datetime and pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if df[c].dtype != "object":
            continue

        raw = df[c].astype(str)

        sample = raw.dropna().head(50)
        if len(sample) >= 10:
            alpha_hits = sum(1 for v in sample if re.search(r"[A-Za-z]", v))
            if alpha_hits / len(sample) >= 0.6:
                continue

        cleaned = (
            raw.str.replace(r"\s+", "", regex=True)
               .str.replace(",", "", regex=False)
               .str.replace("%", "", regex=False)
        )

        num = pd.to_numeric(cleaned, errors="coerce")
        ratio = float(num.notna().mean()) if len(num) else 0.0
        if ratio >= 0.8:
            df[c] = num
            converted.append(str(c))
    return df, converted


def safe_filename_from_path(p: Path, data_dir: Path) -> str:
    try:
        rel = p.relative_to(data_dir)
        s = str(rel).replace(os.sep, "__")
    except Exception:
        s = p.name
    s = re.sub(r"[^\w\-.()]+", "_", s)
    return s


def _is_separator_mismatch_onecol(raw_lines: List[str], sep: str) -> bool:
    seps = [",", "\t", ";", "|"]
    counts = _count_separators_in_lines(raw_lines, seps, max_lines=30)
    total = sum(counts.values())
    if total == 0:
        return False
    max_sep = max(counts, key=lambda k: counts[k])
    if max_sep != sep and counts[max_sep] >= 30 and counts.get(sep, 0) <= 2:
        return True
    return False


def try_read_table(file_path: Path,
                   encodings: List[str],
                   seps_to_try: List[str],
                   max_preview_lines: int) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    meta: Dict[str, str] = {"encoding": "", "sep": "", "skiprows": "", "status": "", "error": "", "header_sep_hint": ""}

    raw_text = None
    chosen_encoding = None
    for enc in encodings:
        try:
            raw_text = file_path.read_text(encoding=enc, errors="strict")
            chosen_encoding = enc
            break
        except Exception:
            continue

    if raw_text is None:
        meta["status"] = "failed"
        meta["error"] = "Could not decode with tried encodings."
        return None, meta

    raw_lines = raw_text.splitlines()
    skiprows, header, sep_hint = detect_skiprows_and_header(raw_lines, seps_to_try, max_preview_lines)
    meta["header_sep_hint"] = repr(sep_hint) if sep_hint else ""

    # 1) sep auto-detect
    try:
        df = pd.read_csv(
            file_path,
            encoding=chosen_encoding,
            sep=None,
            skiprows=skiprows,
            header=header,
            engine="python",
        )
        if df.shape[1] == 1 and _is_separator_mismatch_onecol(raw_lines, sep=","):
            raise ValueError("Auto-sep likely failed (one-column mismatch).")
        meta.update({"encoding": chosen_encoding, "sep": "auto(None)", "skiprows": str(skiprows), "status": "ok"})
        return df, meta

    except Exception as e:
        # ---- RETRY: if skiprows/header detection consumed the file ----
        msg = str(e)
        if "No columns to parse from file" in msg:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=chosen_encoding,
                    sep=None,
                    skiprows=0,
                    header=0,
                    engine="python",
                )
                if df.shape[1] > 0:
                    meta.update({"encoding": chosen_encoding, "sep": "auto(None)", "skiprows": "0", "status": "ok"})
                    return df, meta
            except Exception:
                pass
        pass

    last_err = ""
    # 2) explicit separators
    for sep in seps_to_try:
        try:
            df = pd.read_csv(
                file_path,
                encoding=chosen_encoding,
                sep=sep,
                skiprows=skiprows,
                header=header,
                engine="python",
            )

            if df.shape[1] == 1:
                if _is_separator_mismatch_onecol(raw_lines, sep=sep if isinstance(sep, str) else ","):
                    continue

            meta.update({"encoding": chosen_encoding, "sep": repr(sep), "skiprows": str(skiprows), "status": "ok"})
            return df, meta
        except Exception as e:
            last_err = str(e)

    meta["status"] = "failed"
    meta["error"] = last_err or "read_csv failed"
    meta["encoding"] = chosen_encoding or ""
    meta["skiprows"] = str(skiprows)
    return None, meta


# =========================
# Main
# =========================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic data organizer: read messy CSV/TXT/TSV, clean, normalize, and export cleaned CSV + reports."
    )
    parser.add_argument("--data-dir", type=str, default="DATA_FOLDER", help="Input directory (default: DATA_FOLDER next to script)")
    parser.add_argument("--out-dir", type=str, default="OUTPUT", help="Output directory (default: OUTPUT next to script)")
    parser.add_argument("--extensions", type=str, default="csv,txt,tsv", help="Comma-separated extensions (default: csv,txt,tsv)")
    parser.add_argument("--recursive", action="store_true", help="Search files recursively under data-dir")
    parser.add_argument("--max-preview-lines", type=int, default=DEFAULT_MAX_PREVIEW_LINES, help="Max preview lines for header/skiprows detection")
    parser.add_argument("--no-datetime", action="store_true", help="Disable datetime parsing")
    parser.add_argument("--no-numeric", action="store_true", help="Disable numeric coercion for object columns")
    parser.add_argument("--encodings", type=str, default=",".join(DEFAULT_ENCODINGS), help="Comma-separated encodings to try")
    parser.add_argument("--seps", type=str, default=",".join([",", r"\t", ";", r"\s*\|\s*", "|"]),
                        help=r"Comma-separated separators to try. Use \t for tab. Regex allowed (python engine).")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir

    exts = [x.strip() for x in args.extensions.split(",") if x.strip()]
    encodings = [x.strip() for x in args.encodings.split(",") if x.strip()]
    seps_raw = [x.strip() for x in args.seps.split(",") if x.strip()]
    seps: List[str] = []
    for s in seps_raw:
        if s == r"\t":
            seps.append("\t")
        else:
            seps.append(s)

    clean_dir, meta_dir = ensure_dirs(out_dir)

    if not data_dir.exists():
        print(f"[ERROR] data-dir not found: {data_dir}")
        print("Create the folder and put your data files inside.")
        return 1

    files = iter_target_files(data_dir, exts, recursive=args.recursive)
    if not files:
        print("[INFO] No target files found.")
        return 0

    summary_rows: List[Dict[str, object]] = []
    col_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    metadata_rows_all: List[Dict[str, object]] = []

    for fp in files:
        df, meta = try_read_table(fp, encodings=encodings, seps_to_try=seps, max_preview_lines=args.max_preview_lines)

        row: Dict[str, object] = {
            "file": str(fp.relative_to(data_dir)),
            "status": meta.get("status", ""),
            "encoding": meta.get("encoding", ""),
            "sep": meta.get("sep", ""),
            "skiprows": meta.get("skiprows", ""),
            "header_sep_hint": meta.get("header_sep_hint", ""),
            "n_rows_raw": "",
            "n_cols_raw": "",
            "n_rows_clean": "",
            "n_cols_clean": "",
            "datetime_parsed_cols": "",
            "numeric_converted_cols": "",
            "meta_items_count": 0,
            "error": meta.get("error", ""),
        }

        # ---- metadata extraction (even if df failed, we may still attempt if encoding known) ----
        meta_items: List[Dict[str, str]] = []
        try:
            if row["encoding"]:
                raw_text = fp.read_text(encoding=str(row["encoding"]), errors="ignore")
                raw_lines = raw_text.splitlines()
                header_idx = int(row["skiprows"]) if str(row["skiprows"]).isdigit() else 0
                meta_items = extract_metadata(raw_lines, header_idx=header_idx, seps_to_try=seps)
        except Exception:
            meta_items = []

        row["meta_items_count"] = int(len(meta_items))

        # write per-file metadata csv
        if meta_items:
            out_name = safe_filename_from_path(fp, data_dir=data_dir)
            meta_csv = meta_dir / f"{out_name}.metadata.csv"
            pd.DataFrame(meta_items).to_csv(meta_csv, index=False, encoding="utf-8-sig")

            for m in meta_items:
                metadata_rows_all.append({
                    "file": row["file"],
                    "line_no": m.get("line_no", ""),
                    "key": m.get("key", ""),
                    "value": m.get("value", ""),
                    "raw": m.get("raw", "")
                })

        if df is None:
            summary_rows.append(row)
            failures.append({"file": row["file"], "error": row["error"], "encoding": row["encoding"], "skiprows": row["skiprows"]})
            continue

        row["n_rows_raw"] = int(df.shape[0])
        row["n_cols_raw"] = int(df.shape[1])

        # Cleaning
        df.columns = normalize_columns(list(df.columns))
        df = drop_empty(df)

        parsed_cols: List[str] = []
        if not args.no_datetime:
            df, parsed_cols = try_parse_datetime_columns(df)
        row["datetime_parsed_cols"] = ";".join(parsed_cols)

        converted_cols: List[str] = []
        if not args.no_numeric:
            df, converted_cols = coerce_numeric_columns(df, protect_datetime=True)
        row["numeric_converted_cols"] = ";".join(converted_cols)

        row["n_rows_clean"] = int(df.shape[0])
        row["n_cols_clean"] = int(df.shape[1])

        # Columns inventory
        for c in df.columns:
            col_rows.append(
                {
                    "file": str(fp.relative_to(data_dir)),
                    "column": str(c),
                    "dtype": str(df[c].dtype),
                    "non_null_ratio": float(df[c].notna().mean()) if len(df) else 0.0,
                    "n_unique": int(df[c].nunique(dropna=True)) if len(df) else 0,
                }
            )

        # Save cleaned CSV
        out_name = safe_filename_from_path(fp, data_dir=data_dir)
        out_csv = clean_dir / f"{out_name}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        summary_rows.append(row)

    # Write reports
    pd.DataFrame(summary_rows).to_csv(out_dir / "report_summary.csv", index=False, encoding="utf-8-sig")

    if col_rows:
        pd.DataFrame(col_rows).to_csv(out_dir / "columns_inventory.csv", index=False, encoding="utf-8-sig")

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "report_failures.csv", index=False, encoding="utf-8-sig")

    if metadata_rows_all:
        pd.DataFrame(metadata_rows_all).to_csv(out_dir / "report_metadata.csv", index=False, encoding="utf-8-sig")

    print(f"[DONE] Processed {len(files)} files.")
    print(f" - Cleaned CSV: {clean_dir}")
    print(f" - Metadata:    {meta_dir}")
    print(f" - Summary:     {out_dir / 'report_summary.csv'}")
    if metadata_rows_all:
        print(f" - Meta report: {out_dir / 'report_metadata.csv'}")
    if failures:
        print(f" - Failures:    {out_dir / 'report_failures.csv'}")
    print(f" - Columns:     {out_dir / 'columns_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
