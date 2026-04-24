"""
Cognitive Symmetry: Main Pipeline
=================================
GPT-2 × Dundee Corpus (Mid-Submission)

Runs the full analysis pipeline:
1. Load Dundee corpus (multiple texts, all subjects)
2. Classify sentence structures (active/passive/garden_path/relative_clause)
3. Run GPT-2 forward passes → surprisal + attention entropy
4. Align human eye-tracking data with LLM metrics
5. Correlation analysis (Pearson r)
6. Saccadic heads identification (layer × head heatmap)
7. Regression-entropy lag analysis
8. Sentence structure comparison
9. Layer-to-brain-part mapping
10. Bayesian-style regression (statsmodels)
11. Generate comprehensive report
"""

import os
import sys
from tqdm import tqdm

from data_utils import load_dundee_dataset, preprocess_for_model
from sentence_structure import classify_sentences_batch
from llm_pipeline import GPT2Evaluator
from analysis import (
    align_and_correlate,
    plot_saccadic_heads,
    analyze_regression_entropy_lag,
    compare_sentence_structures,
    analyze_layer_brain_mapping,
    plot_reading_time_effects,
    plot_surprisal_vs_entropy,
    save_comprehensive_report,
)
from bayesian_analysis import run_logistic_regression, run_linear_regression


# ── Configuration ──
NUM_DUNDEE_TEXTS = 5          # How many Dundee text files to parse (1–20)
MAX_SENTENCES_PER_TEXT = 100  # Max sentences per text file
MAX_SENTENCES_TO_EVAL = 200  # Max sentences to run through GPT-2


def main():
    print("=" * 70)
    print("  COGNITIVE SYMMETRY PIPELINE")
    print("  Model: GPT-2 (Low-Tier)  |  Corpus: Dundee  |  Mid-Submission")
    print("=" * 70)

    # ── 1. Create output directories ──
    os.makedirs("results/images", exist_ok=True)
    os.makedirs("results/texts", exist_ok=True)
    os.makedirs("results/bayesian", exist_ok=True)

    # ── 2. Load Dundee corpus ──
    print("\n[Step 1/7] Loading Dundee Corpus...")
    dundee_df = load_dundee_dataset(
        "dundee",
        num_texts=NUM_DUNDEE_TEXTS,
        max_sentences_per_text=MAX_SENTENCES_PER_TEXT
    )

    if dundee_df.empty:
        print("ERROR: No Dundee data loaded. Exiting.")
        sys.exit(1)

    # ── 3. Classify sentence structures ──
    print("\n[Step 2/7] Classifying sentence structures...")
    unique_sentences = dundee_df["sentence"].unique().tolist()
    sentence_labels = classify_sentences_batch(unique_sentences)

    # Add labels to DataFrame
    dundee_df["sentence_structure"] = dundee_df["sentence"].map(sentence_labels)

    # ── 4. Prepare sentences for LLM ──
    print("\n[Step 3/7] Preparing data for GPT-2...")
    human_data = preprocess_for_model(dundee_df)
    print(f"  Total sentences: {len(human_data)}")

    # Limit for computational feasibility
    if len(human_data) > MAX_SENTENCES_TO_EVAL:
        print(f"  Limiting to first {MAX_SENTENCES_TO_EVAL} sentences for evaluation.")
        human_data = human_data[:MAX_SENTENCES_TO_EVAL]

    # ── 5. Load GPT-2 and evaluate ──
    print("\n[Step 4/7] Running GPT-2 evaluation...")
    evaluator = GPT2Evaluator("gpt2")

    llm_data_metrics = []
    for item in tqdm(human_data, desc="GPT-2 Processing"):
        sentence = item["sentence"]
        try:
            metrics = evaluator.evaluate_sentence(sentence)
            llm_data_metrics.append({
                "sentence": sentence,
                "metrics": metrics,
            })
        except Exception as e:
            print(f"  [WARN] Skipping sentence (error: {e})")
            continue

    print(f"  Successfully evaluated {len(llm_data_metrics)} / {len(human_data)} sentences.")

    # Trim human data to match
    aligned_human = human_data[:len(llm_data_metrics)]

    # ── 6. Align and correlate ──
    print("\n[Step 5/7] Aligning human + LLM data and computing correlations...")
    df, corrs = align_and_correlate(aligned_human, llm_data_metrics, sentence_labels)

    if df.empty:
        print("ERROR: No aligned data. Exiting.")
        sys.exit(1)

    print(f"  Aligned tokens: {len(df)}")

    # ── 7. Run all analyses ──
    print("\n[Step 6/7] Running analyses...")
    model_tag = "GPT2"
    dataset_tag = "DUNDEE"
    combined_tag = f"{model_tag}_{dataset_tag}"

    # 7a. Saccadic heads
    print("\n── Saccadic Heads ──")
    head_corrs = plot_saccadic_heads(df, combined_tag)

    # 7b. Regression-Entropy Lag
    print("\n── Regression-Entropy Lag ──")
    analyze_regression_entropy_lag(df, combined_tag)

    # 7c. Sentence Structure Comparison
    print("\n── Sentence Structure Comparison ──")
    compare_sentence_structures(df, combined_tag)

    # 7d. Layer-to-Brain Mapping
    print("\n── Layer-to-Brain Mapping ──")
    analyze_layer_brain_mapping(df, combined_tag, num_layers=12)

    # 7e. Reading Time Effects
    print("\n── Reading Time Effects ──")
    plot_reading_time_effects(df, combined_tag)

    # 7f. Surprisal-Entropy scatter
    print("\n── Surprisal-Entropy Scatter ──")
    plot_surprisal_vs_entropy(df, combined_tag)

    # 7g. Bayesian-style regressions
    print("\n── Regression Modelling ──")
    run_logistic_regression(df, output_dir="results/bayesian")
    run_linear_regression(df, output_dir="results/bayesian")

    # ── 8. Save comprehensive report ──
    print("\n[Step 7/7] Generating comprehensive report...")
    save_comprehensive_report(df, corrs, head_corrs, model_tag, dataset_tag)

    # Save aligned DataFrame for further analysis
    df.to_csv("results/aligned_data_GPT2_DUNDEE.csv", index=False)
    print(f"  Saved aligned data → results/aligned_data_GPT2_DUNDEE.csv")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print(f"  Results: results/images/, results/texts/, results/bayesian/")
    print("=" * 70)


if __name__ == "__main__":
    main()
