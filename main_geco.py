import argparse
import os
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from data_utils import load_dundee_dataset, preprocess_for_model
from llm_pipeline import GPT2Evaluator, BertEvaluator
from analysis import (
    align_and_correlate,
    plot_saccadic_heads,
    plot_regression_entropy_lag,
    save_text_results,
    plot_reading_time_effects,
)

MISSING_VALUES = {".", "", "nan", "NaN", "None", "none", "NA", "N/A"}
SENT_END_RE = re.compile(r"[.!?][\"')\]]*$")


def _read_table(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"GECO file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls", ".xlsm"}:
        try:
            return pd.read_excel(path, sheet_name=sheet_name or "DATA")
        except ValueError:
            return pd.read_excel(path, sheet_name=0)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _normalise_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace(list(MISSING_VALUES), np.nan, inplace=True)
    return df.infer_objects(copy=False)


def _find_by_patterns(df: pd.DataFrame, patterns: Sequence[str], exclude: Sequence[str] = ()) -> Optional[str]:
    for pattern in patterns:
        rx = re.compile(pattern, flags=re.I)
        ex = [re.compile(e, flags=re.I) for e in exclude]
        for c in df.columns:
            name = str(c)
            if rx.search(name) and not any(e.search(name) for e in ex):
                return c
    return None


def _choose_word_col(df: pd.DataFrame, override: Optional[str] = None) -> str:
    if override:
        return override

    explicit = _find_by_patterns(df, [r"^word$", r"word(_|\s)*(form|text|token)$", r"^token$"])
    if explicit:
        return explicit

    meta_rx = re.compile(r"pp|participant|subject|group|language|trial|part|id|nr", re.I)
    best_col, best_score = None, -1.0
    for c in df.columns:
        if meta_rx.search(str(c)):
            continue
        s = df[c].dropna().astype(str).head(1000)
        if s.empty:
            continue
        alpha_frac = s.str.contains(r"[A-Za-z]", regex=True).mean()
        numeric_frac = s.str.match(r"^-?\d+(\.\d+)?$", na=False).mean()
        id_frac = s.str.match(r"^\d+[-_.]\d+[-_.]?\d*$", na=False).mean()
        avg_len = s.str.len().mean()
        unique_frac = s.nunique() / max(len(s), 1)
        score = alpha_frac * 3.0 + unique_frac - numeric_frac * 3.0 - id_frac * 2.0 - max(avg_len - 18, 0) * 0.05
        if score > best_score:
            best_col, best_score = c, score
    if best_col is None:
        raise ValueError("Could not detect GECO word/token column. Pass --geco-word-col explicitly.")
    return best_col


def _choose_word_id_col(df: pd.DataFrame, word_col: str, override: Optional[str] = None) -> Optional[str]:
    if override:
        return override

    # Prefer full GECO IDs such as 1-5-14 if present.
    candidate_cols = []
    for c in df.columns:
        name = str(c)
        if c == word_col:
            continue
        if re.search(r"word.*id|token.*id|ia.*id", name, flags=re.I):
            candidate_cols.append(c)

    best_col, best_score = None, -1.0
    for c in candidate_cols:
        s = df[c].dropna().astype(str).head(1000).str.strip()
        if s.empty:
            continue
        triple_frac = s.str.match(r"^\d+[-_.]\d+[-_.]\d+$", na=False).mean()
        numeric_frac = s.str.match(r"^\d+(\.0)?$", na=False).mean()
        # Triple IDs are most useful; simple within-trial numeric IDs are also useful.
        score = triple_frac * 3.0 + numeric_frac
        if score > best_score:
            best_col, best_score = c, score
    return best_col


def _choose_part_trial_cols(df: pd.DataFrame):
    part_col = _find_by_patterns(df, [r"^part$", r"^text$", r"text.*id", r"paragraph"])
    trial_col = _find_by_patterns(df, [r"^trial$", r"trial.*id", r"sentence.*id"])
    return part_col, trial_col


def _choose_metric_col(df: pd.DataFrame, override: Optional[str], preferred: Sequence[str], fallback: Sequence[str] = ()) -> Optional[str]:
    if override:
        return override
    # First: word-level metrics only. This avoids accidentally selecting TRIAL_TOTAL_READING_TIME.
    col = _find_by_patterns(df, preferred, exclude=[r"^trial", r"trial_"])
    if col is not None:
        return col
    # Second: safe fallbacks only if no word-level metric exists.
    return _find_by_patterns(df, fallback, exclude=[r"^trial", r"trial_"])


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _metric_from_col(df: pd.DataFrame, col: Optional[str], binary: bool = False) -> pd.Series:
    if col is None:
        return pd.Series(np.zeros(len(df)), index=df.index)
    x = _to_numeric(df[col], default=0.0)
    return (x > 0).astype(float) if binary else x


def _split_trials_into_sentences(agg: pd.DataFrame, max_words_per_segment: int = 180) -> pd.DataFrame:
    """Split GECO trial/page rows into model-sized sentences/chunks."""
    rows = []
    for trial_id, group in agg.groupby("trial_sentence_id", sort=False):
        seg = 0
        local_idx = 0
        words_in_seg = 0
        for _, row in group.sort_values("word_idx", kind="stable").iterrows():
            word = str(row["word"])
            out = row.to_dict()
            out["sentence_id"] = f"{trial_id}__S{seg:04d}"
            out["word_idx"] = local_idx
            rows.append(out)

            words_in_seg += 1
            local_idx += 1
            if SENT_END_RE.search(word) or words_in_seg >= max_words_per_segment:
                seg += 1
                local_idx = 0
                words_in_seg = 0

    split = pd.DataFrame(rows)
    sentence_map = split.groupby("sentence_id", sort=False)["word"].apply(lambda x: " ".join(x.astype(str))).to_dict()
    split["sentence"] = split["sentence_id"].map(sentence_map)
    return split[["sentence", "word_idx", "word", "regression_in_prob", "regression_out_prob", "reading_time"]]


def load_geco_word_level(
    filepath: str,
    sheet_name: Optional[str] = None,
    word_col: Optional[str] = None,
    word_id_col: Optional[str] = None,
    reading_time_col: Optional[str] = None,
    regression_in_col: Optional[str] = None,
    regression_out_col: Optional[str] = None,
    max_sentences: Optional[int] = None,
    max_words_per_segment: int = 180,
) -> pd.DataFrame:
    raw = _normalise_missing(_read_table(filepath, sheet_name=sheet_name))

    word_col = _choose_word_col(raw, word_col)
    word_id_col = _choose_word_id_col(raw, word_col, word_id_col)
    part_col, trial_col = _choose_part_trial_cols(raw)

    reading_time_col = _choose_metric_col(raw, reading_time_col, [
        r"word.*total.*reading.*time", r"word.*reading.*time", r"word.*dwell.*time",
        r"word.*total.*fixation.*duration", r"word.*gaze.*duration", r"word.*go[-_ ]?past",
        r"word.*regression.*path", r"^trt$", r"^gpt$", r"^ffd$",
    ], fallback=[r"total.*reading.*time", r"reading.*time", r"dwell.*time", r"gaze.*duration"])

    regression_in_col = _choose_metric_col(raw, regression_in_col, [
        r"word.*regression.*in", r"word.*reg.*in", r"word.*incoming.*reg", r"word.*refix", r"word.*re[-_ ]?read",
        r"regression.*in", r"reg.*in", r"incoming.*reg", r"refix", r"re[-_ ]?read",
    ])
    regression_out_col = _choose_metric_col(raw, regression_out_col, [
        r"word.*regression.*out", r"word.*reg.*out", r"word.*outgoing.*reg", r"regression.*out", r"reg.*out", r"outgoing.*reg",
    ])

    df = raw.copy()
    df["word"] = df[word_col].astype(str).str.strip()
    df = df[df["word"].notna() & (df["word"] != "") & (df["word"].str.lower() != "nan")]

    # Correct GECO grouping:
    # - If WORD_ID looks like 1-5-14, use first two parts as trial/page and third as word index.
    # - If WORD_ID_WITHIN_TRIAL is just 1,2,3,... use PART/TRIAL as trial/page and the numeric value as word index.
    if word_id_col is not None:
        wid = df[word_id_col].astype(str).str.strip()
        triple = wid.str.extract(r"^(\d+)[-_.](\d+)[-_.](\d+)$")
        if triple.notna().all(axis=1).mean() > 0.50:
            df["trial_sentence_id"] = triple[0].astype(str) + "-" + triple[1].astype(str)
            df["word_idx"] = pd.to_numeric(triple[2], errors="coerce")
        else:
            simple_num = pd.to_numeric(wid, errors="coerce")
            if simple_num.notna().mean() > 0.50:
                group_cols = [c for c in [part_col, trial_col] if c is not None]
                if group_cols:
                    df["trial_sentence_id"] = df[group_cols].astype(str).agg("-".join, axis=1)
                else:
                    # Last resort: keep one long group, then chunk it safely.
                    df["trial_sentence_id"] = "GECO_1"
                df["word_idx"] = simple_num
            else:
                df["trial_sentence_id"] = wid
                df["word_idx"] = np.nan
    else:
        group_cols = [c for c in [part_col, trial_col] if c is not None]
        df["trial_sentence_id"] = df[group_cols].astype(str).agg("-".join, axis=1) if group_cols else "GECO_1"
        df["word_idx"] = np.nan

    if df["word_idx"].isna().all():
        df["word_idx"] = df.groupby("trial_sentence_id").cumcount() + 1
    else:
        df["word_idx"] = pd.to_numeric(df["word_idx"], errors="coerce")
        df["word_idx"] = df.groupby("trial_sentence_id")["word_idx"].transform(
            lambda x: x.fillna(pd.Series(range(1, len(x) + 1), index=x.index))
        )

    df["reading_time_raw"] = _metric_from_col(df, reading_time_col, binary=False)
    df["regression_in_raw"] = _metric_from_col(df, regression_in_col, binary=True)
    df["regression_out_raw"] = _metric_from_col(df, regression_out_col, binary=True)

    if regression_in_col is None:
        print("[GECO] Warning: no explicit Regression-In column detected. Values will be 0 unless you pass --geco-regression-in-col.")
    if regression_out_col is None:
        print("[GECO] Warning: no explicit Regression-Out column detected. Values will be 0 unless you pass --geco-regression-out-col.")
    if reading_time_col is None:
        print("[GECO] Warning: no word-level reading-time column detected. Values will be 0 unless you pass --geco-reading-time-col.")

    agg = (
        df.groupby(["trial_sentence_id", "word_idx", "word"], sort=False)
        .agg(
            regression_in_prob=("regression_in_raw", "mean"),
            regression_out_prob=("regression_out_raw", "mean"),
            reading_time=("reading_time_raw", lambda s: float(np.nanmean(s.replace(0, np.nan))) if (s > 0).any() else 0.0),
        )
        .reset_index()
        .sort_values(["trial_sentence_id", "word_idx"], kind="stable")
    )

    final = _split_trials_into_sentences(agg, max_words_per_segment=max_words_per_segment)

    if max_sentences is not None:
        keep_ids = list(final["sentence"].drop_duplicates().head(max_sentences))
        final = final[final["sentence"].isin(keep_ids)]

    print("[GECO] Column mapping used:")
    print(f"  word_col            = {word_col}")
    print(f"  word_id_col         = {word_id_col}")
    print(f"  part_col            = {part_col}")
    print(f"  trial_col           = {trial_col}")
    print(f"  reading_time_col    = {reading_time_col}")
    print(f"  regression_in_col   = {regression_in_col}")
    print(f"  regression_out_col  = {regression_out_col}")
    print(f"[GECO] Normalized to {len(final)} word rows across {final['sentence'].nunique()} sentence/chunk units.")

    return final.reset_index(drop=True)


def process_dataset(dataset_name, dataset_df, evaluators, max_sentences: Optional[int] = None):
    print(f"=== Processing [{dataset_name}] ===")
    human_data = preprocess_for_model(dataset_df)
    if max_sentences is not None:
        human_data = human_data[:max_sentences]

    for evaluator_name, evaluator in evaluators.items():
        print(f"\n  -> Running {evaluator_name} on {dataset_name}...")
        llm_data_metrics = []
        for i, item in enumerate(human_data, start=1):
            sentence = item["sentence"]
            try:
                metrics = evaluator.evaluate_sentence(sentence)
                llm_data_metrics.append({"sentence": sentence, "metrics": metrics})
            except Exception as e:
                print(f"[WARN] Skipping sentence/chunk {i} because model evaluation failed: {e}")

        print(f"  -> Aligning and correlating {evaluator_name} vs {dataset_name}...")
        aligned_human = human_data[:len(llm_data_metrics)]
        df, corrs = align_and_correlate(aligned_human, llm_data_metrics)

        save_text_results(corrs, evaluator_name, dataset_name)
        plot_saccadic_heads(df, f"{evaluator_name}_{dataset_name}")
        plot_regression_entropy_lag(df, f"{evaluator_name}_{dataset_name}")
        plot_reading_time_effects(df, f"{evaluator_name}_{dataset_name}")
        print("-" * 40)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cognitive Symmetry on Dundee and/or GECO.")
    parser.add_argument("--dataset", choices=["dundee", "geco", "both"], default="both")
    parser.add_argument("--dundee-path", default="dundee")
    parser.add_argument("--geco-path", default="data/geco.xlsx")
    parser.add_argument("--geco-sheet", default=None)
    parser.add_argument("--max-sentences", type=int, default=20, help="Limit sentence/chunks for quick runs. Use 0 for full GECO.")
    parser.add_argument("--max-words-per-segment", type=int, default=180, help="Safety chunk size before feeding GPT-2/BERT.")
    parser.add_argument("--model", choices=["gpt2", "bert", "both"], default="both")

    parser.add_argument("--geco-word-col", default=None)
    parser.add_argument("--geco-word-id-col", default=None)
    parser.add_argument("--geco-reading-time-col", default=None)
    parser.add_argument("--geco-regression-in-col", default=None)
    parser.add_argument("--geco-regression-out-col", default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    max_sentences = None if args.max_sentences == 0 else args.max_sentences

    print("Initializing Cognitive Symmetry Pipeline...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    evaluators = {}
    if args.model in {"gpt2", "both"}:
        evaluators["GPT2"] = GPT2Evaluator("gpt2")
    if args.model in {"bert", "both"}:
        evaluators["BERT"] = BertEvaluator("bert-base-uncased")

    if args.dataset in {"dundee", "both"}:
        dundee_df = load_dundee_dataset(args.dundee_path)
        process_dataset("DUNDEE", dundee_df, evaluators, max_sentences=max_sentences)

    if args.dataset in {"geco", "both"}:
        geco_df = load_geco_word_level(
            filepath=args.geco_path,
            sheet_name=args.geco_sheet,
            word_col=args.geco_word_col,
            word_id_col=args.geco_word_id_col,
            reading_time_col=args.geco_reading_time_col,
            regression_in_col=args.geco_regression_in_col,
            regression_out_col=args.geco_regression_out_col,
            max_sentences=max_sentences,
            max_words_per_segment=args.max_words_per_segment,
        )
        process_dataset("GECO", geco_df, evaluators, max_sentences=max_sentences)

    print("Pipeline Complete! Check the 'results' folder for output artifacts.")


if __name__ == "__main__":
    main()
