"""
Comprehensive analysis module for Cognitive Symmetry project.

Includes:
1. Pearson correlation analysis (surprisal / entropy vs regression metrics)
2. Saccadic heads identification (layer×head heatmap + top-k ranking)
3. Regression-Entropy Lag analysis (temporal alignment)
4. Sentence structure grouped comparison
5. Layer-to-brain-part mapping (early/middle/late → visual/syntactic/semantic)
6. Reading time effects
7. Comprehensive text report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Style config ──
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.family': 'sans-serif',
})


# ============================================================
# 1. Core alignment and correlation
# ============================================================

def align_and_correlate(human_data, llm_data, sentence_labels=None):
    """
    Align human eye-tracking data with LLM metrics word-by-word.

    human_data: list of {"sentence", "metrics": [{"word", "regression_in_prob", ...}]}
    llm_data: list of {"sentence", "metrics": [{"word", "surprisal", "attention_entropy": {}}]}
    sentence_labels: optional dict mapping sentence → structure label

    Returns: (aligned_df, correlations_dict)
    """
    results = []

    for h_sent, l_sent in zip(human_data, llm_data):
        h_metrics = h_sent["metrics"]
        l_metrics = l_sent["metrics"]
        sent_text = h_sent["sentence"]

        for h_w, l_w in zip(h_metrics, l_metrics):
            if h_w["word"].lower().rstrip(".,!?;:") == l_w["word"].lower().rstrip(".,!?;:"):
                row = {
                    "sentence": sent_text,
                    "word": h_w["word"],
                    "word_idx": h_w["word_idx"],
                    "regression_in": h_w["regression_in_prob"],
                    "regression_out": h_w["regression_out_prob"],
                    "reading_time": h_w.get("reading_time", 0),
                    "surprisal": l_w["surprisal"],
                    "text_id": h_w.get("text_id", 0),
                    "sentence_id": h_w.get("sentence_id_global", 0),
                }

                # Per-head entropies
                for k, v in l_w["attention_entropy"].items():
                    row[f"entropy_{k}"] = v

                # Mean entropy across all heads
                if len(l_w["attention_entropy"]) > 0:
                    row["entropy_mean"] = np.mean(list(l_w["attention_entropy"].values()))
                else:
                    row["entropy_mean"] = 0

                # Sentence structure label
                if sentence_labels and sent_text in sentence_labels:
                    row["sentence_structure"] = sentence_labels[sent_text]

                results.append(row)

    df = pd.DataFrame(results)

    if len(df) < 2:
        print("[align] Warning: fewer than 2 aligned words.")
        return df, {}

    # ── Compute correlations ──
    corrs = _compute_correlations(df)
    _print_correlations(corrs)

    return df, corrs


def _compute_correlations(df):
    """Compute all key Pearson correlations."""
    corrs = {}

    def safe_pearsonr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return 0.0, 1.0
        return pearsonr(x[mask], y[mask])

    # Core correlations
    r, p = safe_pearsonr(df["entropy_mean"], df["regression_in"])
    corrs["entropy_vs_reg_in"] = {"r": r, "p": p}

    r, p = safe_pearsonr(df["entropy_mean"], df["regression_out"])
    corrs["entropy_vs_reg_out"] = {"r": r, "p": p}

    r, p = safe_pearsonr(df["surprisal"], df["regression_in"])
    corrs["surprisal_vs_reg_in"] = {"r": r, "p": p}

    r, p = safe_pearsonr(df["surprisal"], df["regression_out"])
    corrs["surprisal_vs_reg_out"] = {"r": r, "p": p}

    if "reading_time" in df.columns and df["reading_time"].sum() > 0:
        r, p = safe_pearsonr(df["reading_time"], df["entropy_mean"])
        corrs["rt_vs_entropy"] = {"r": r, "p": p}
        r, p = safe_pearsonr(df["reading_time"], df["surprisal"])
        corrs["rt_vs_surprisal"] = {"r": r, "p": p}

    return corrs


def _print_correlations(corrs):
    """Pretty-print correlation results."""
    print("\n" + "=" * 60)
    print("PEARSON CORRELATIONS")
    print("=" * 60)
    for name, vals in corrs.items():
        sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else "ns"
        print(f"  {name:30s}: r = {vals['r']:+.4f}  p = {vals['p']:.2e}  {sig}")
    print("=" * 60 + "\n")


# ============================================================
# 2. Saccadic Heads Identification
# ============================================================

def plot_saccadic_heads(df, model_name, output_dir="results/images"):
    """
    Heatmap of per-head correlation with regression-in.
    Also identifies top-k 'saccadic heads'.
    """
    os.makedirs(output_dir, exist_ok=True)

    entropy_cols = [c for c in df.columns if c.startswith("entropy_L")]
    if not entropy_cols:
        print("[SaccadicHeads] No entropy columns found.")
        return []

    head_corrs = []
    for col in entropy_cols:
        mask = np.isfinite(df[col]) & np.isfinite(df["regression_in"])
        if mask.sum() < 3:
            continue
        r, p = pearsonr(df.loc[mask, col], df.loc[mask, "regression_in"])
        parts = col.split("_")
        layer = int(parts[1][1:])
        head = int(parts[2][1:])
        head_corrs.append({"Layer": layer, "Head": head, "Correlation": r, "p_value": p})

    head_df = pd.DataFrame(head_corrs)
    if head_df.empty:
        return []

    # ── Heatmap ──
    pivot_df = head_df.pivot(index="Layer", columns="Head", values="Correlation")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title(f"Saccadic Heads: Correlation with Human Regression-In\n({model_name})",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Attention Head", fontsize=12)
    ax.set_ylabel("Transformer Layer", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/saccadic_heads_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Top saccadic heads ──
    top_heads = head_df.nlargest(10, "Correlation")
    print(f"\n  Top 10 Saccadic Heads ({model_name}):")
    for _, row in top_heads.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else \
              "*" if row["p_value"] < 0.05 else ""
        print(f"    L{int(row['Layer']):2d} H{int(row['Head']):2d}: "
              f"r = {row['Correlation']:+.4f}  p = {row['p_value']:.2e} {sig}")

    print(f"  Saved heatmap → {output_dir}/saccadic_heads_{model_name}.png")
    return head_corrs


# ============================================================
# 3. Regression-Entropy Lag Analysis
# ============================================================

def analyze_regression_entropy_lag(df, model_name, output_dir="results/images"):
    """
    Investigates whether LLM entropy spikes AT the high-surprisal word,
    or 1-2 tokens AFTER it (the 'lag' phenomenon).

    For each sentence, finds the peak-surprisal token and checks the
    entropy at offsets [-1, 0, +1, +2].
    """
    os.makedirs(output_dir, exist_ok=True)

    lag_data = []

    for sent, group in df.groupby("sentence"):
        group = group.sort_values("word_idx").reset_index(drop=True)
        if len(group) < 4:
            continue

        # Find the peak-surprisal word (exclude first word)
        surp_values = group["surprisal"].values
        if len(surp_values) < 3:
            continue
        peak_idx = np.argmax(surp_values[1:]) + 1  # skip first token

        # Check entropy at offsets around the peak
        for offset in [-2, -1, 0, 1, 2]:
            check_idx = peak_idx + offset
            if 0 <= check_idx < len(group):
                lag_data.append({
                    "offset": offset,
                    "entropy": group.iloc[check_idx]["entropy_mean"],
                    "surprisal": group.iloc[check_idx]["surprisal"],
                    "regression_in": group.iloc[check_idx]["regression_in"],
                    "sentence": sent,
                })

    lag_df = pd.DataFrame(lag_data)
    if lag_df.empty:
        print("[Lag] No lag data computed.")
        return

    # ── Plot 1: Mean entropy by offset ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    agg = lag_df.groupby("offset").agg(
        entropy_mean=("entropy", "mean"),
        entropy_sem=("entropy", "sem"),
        reg_in_mean=("regression_in", "mean"),
        reg_in_sem=("regression_in", "sem"),
    ).reset_index()

    # Entropy subplot
    axes[0].bar(agg["offset"], agg["entropy_mean"], yerr=agg["entropy_sem"],
                color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
                alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel("Token Offset from Peak Surprisal", fontsize=11)
    axes[0].set_ylabel("Mean Attention Entropy", fontsize=11)
    axes[0].set_title("LLM Entropy Around High-Surprisal Token", fontsize=12, fontweight='bold')
    axes[0].set_xticks([-2, -1, 0, 1, 2])
    axes[0].set_xticklabels(["-2", "-1", "Peak", "+1", "+2"])
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Regression-in subplot
    axes[1].bar(agg["offset"], agg["reg_in_mean"], yerr=agg["reg_in_sem"],
                color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
                alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel("Token Offset from Peak Surprisal", fontsize=11)
    axes[1].set_ylabel("Mean Regression-In Probability", fontsize=11)
    axes[1].set_title("Human Regression Around High-Surprisal Token", fontsize=12, fontweight='bold')
    axes[1].set_xticks([-2, -1, 0, 1, 2])
    axes[1].set_xticklabels(["-2", "-1", "Peak", "+1", "+2"])
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.suptitle(f"Regression-Entropy Lag Analysis ({model_name})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regression_entropy_lag_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved lag plot → {output_dir}/regression_entropy_lag_{model_name}.png")

    # ── Quantify lag ──
    peak_entropy = agg.loc[agg["offset"] == 0, "entropy_mean"].values[0]
    for off in [1, 2]:
        post_ent = agg.loc[agg["offset"] == off, "entropy_mean"].values
        if len(post_ent) > 0:
            ratio = post_ent[0] / peak_entropy if peak_entropy > 0 else 0
            print(f"  Entropy at offset +{off} / peak: {ratio:.3f}")


# ============================================================
# 4. Sentence Structure Comparison
# ============================================================

def compare_sentence_structures(df, model_name, output_dir="results/images"):
    """
    Grouped bar chart: correlation within each sentence structure class.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "sentence_structure" not in df.columns:
        print("[Structure] No sentence_structure column.")
        return

    structures = df["sentence_structure"].unique()
    if len(structures) < 2:
        print(f"[Structure] Only {len(structures)} structure(s) found, skipping comparison.")
        return

    results = []
    for struct in sorted(structures):
        sub = df[df["sentence_structure"] == struct]
        if len(sub) < 5:
            continue

        def safe_r(x, y):
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                return 0.0
            return pearsonr(x[m], y[m])[0]

        results.append({
            "Structure": struct,
            "N_words": len(sub),
            "N_sentences": sub["sentence"].nunique(),
            "Surprisal↔RegIn": safe_r(sub["surprisal"], sub["regression_in"]),
            "Entropy↔RegIn": safe_r(sub["entropy_mean"], sub["regression_in"]),
            "Surprisal↔RegOut": safe_r(sub["surprisal"], sub["regression_out"]),
        })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return

    # ── Grouped bar chart ──
    metrics_to_plot = ["Surprisal↔RegIn", "Entropy↔RegIn", "Surprisal↔RegOut"]
    x = np.arange(len(res_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, res_df[metric], width, label=metric, color=colors[i], alpha=0.8,
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Sentence Structure", fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title(f"Correlation by Sentence Structure ({model_name})", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{r['Structure']}\n(n={r['N_sentences']})" for _, r in res_df.iterrows()])
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/structure_comparison_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved structure comparison → {output_dir}/structure_comparison_{model_name}.png")

    # Print table
    print(f"\n  Structure Comparison ({model_name}):")
    print(res_df.to_string(index=False))


# ============================================================
# 5. Layer-to-Brain-Part Mapping
# ============================================================

def analyze_layer_brain_mapping(df, model_name, num_layers=12, output_dir="results/images"):
    """
    Maps GPT-2's 12 layers into functional brain regions:
      - Early  (L0–L3):  Visual/Orthographic processing (V1/V2, visual word form area)
      - Middle (L4–L7):  Syntactic parsing (Broca's area, left IFG)
      - Late   (L8–L11): Semantic integration (Wernicke's area, angular gyrus, TPJ)

    Computes per-layer-group correlation with human regression-in and reading time.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define the mapping
    brain_map = {
        "Early (L0–L3)\nVisual/Orthographic\n(V1, VWFA)": list(range(0, 4)),
        "Middle (L4–L7)\nSyntactic Parsing\n(Broca's, Left IFG)": list(range(4, 8)),
        "Late (L8–L11)\nSemantic Integration\n(Wernicke's, Angular G.)": list(range(8, 12)),
    }

    # Short labels for printing
    brain_labels_short = {
        "Early (L0–L3)\nVisual/Orthographic\n(V1, VWFA)": "Early (Visual/Ortho.)",
        "Middle (L4–L7)\nSyntactic Parsing\n(Broca's, Left IFG)": "Middle (Syntactic)",
        "Late (L8–L11)\nSemantic Integration\n(Wernicke's, Angular G.)": "Late (Semantic)",
    }

    # ── Compute per-layer correlation with regression-in ──
    layer_results = []
    for layer_idx in range(num_layers):
        # Get all head entropy columns for this layer
        layer_cols = [c for c in df.columns if c.startswith(f"entropy_L{layer_idx}_")]
        if not layer_cols:
            continue

        # Mean entropy across heads in this layer
        layer_entropy = df[layer_cols].mean(axis=1)
        mask = np.isfinite(layer_entropy) & np.isfinite(df["regression_in"])
        if mask.sum() < 3:
            continue

        r_in, p_in = pearsonr(layer_entropy[mask], df.loc[mask, "regression_in"])
        r_out, p_out = pearsonr(layer_entropy[mask], df.loc[mask, "regression_out"])

        r_rt, p_rt = 0.0, 1.0
        if "reading_time" in df.columns and df["reading_time"].sum() > 0:
            rt_mask = mask & np.isfinite(df["reading_time"]) & (df["reading_time"] > 0)
            if rt_mask.sum() > 3:
                r_rt, p_rt = pearsonr(layer_entropy[rt_mask], df.loc[rt_mask, "reading_time"])

        layer_results.append({
            "Layer": layer_idx,
            "r_regression_in": r_in,
            "p_regression_in": p_in,
            "r_regression_out": r_out,
            "p_regression_out": p_out,
            "r_reading_time": r_rt,
            "p_reading_time": p_rt,
        })

    layer_df = pd.DataFrame(layer_results)
    if layer_df.empty:
        print("[LayerMap] No layer data.")
        return

    # ── Grouped brain region correlations ──
    region_results = []
    for region_name, layer_indices in brain_map.items():
        region_cols = []
        for l in layer_indices:
            region_cols.extend([c for c in df.columns if c.startswith(f"entropy_L{l}_")])

        if not region_cols:
            continue

        region_entropy = df[region_cols].mean(axis=1)
        mask = np.isfinite(region_entropy) & np.isfinite(df["regression_in"])
        if mask.sum() < 3:
            continue

        r_in, p_in = pearsonr(region_entropy[mask], df.loc[mask, "regression_in"])
        r_out, _ = pearsonr(region_entropy[mask], df.loc[mask, "regression_out"])

        r_rt = 0.0
        if "reading_time" in df.columns and df["reading_time"].sum() > 0:
            rt_mask = mask & np.isfinite(df["reading_time"]) & (df["reading_time"] > 0)
            if rt_mask.sum() > 3:
                r_rt, _ = pearsonr(region_entropy[rt_mask], df.loc[rt_mask, "reading_time"])

        region_results.append({
            "Region": region_name,
            "Region_short": brain_labels_short[region_name],
            "r_regression_in": r_in,
            "p_regression_in": p_in,
            "r_regression_out": r_out,
            "r_reading_time": r_rt,
        })

    region_df = pd.DataFrame(region_results)

    # ============ PLOT: Combined figure ============
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Panel A: Per-layer bar chart (Regression-In) ──
    ax = axes[0, 0]
    colors_by_layer = []
    for l in layer_df["Layer"]:
        if l < 4:
            colors_by_layer.append('#3498db')   # Early - blue
        elif l < 8:
            colors_by_layer.append('#e74c3c')   # Middle - red
        else:
            colors_by_layer.append('#2ecc71')   # Late - green

    ax.bar(layer_df["Layer"], layer_df["r_regression_in"],
           color=colors_by_layer, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("GPT-2 Layer", fontsize=11)
    ax.set_ylabel("Pearson r (vs Regression-In)", fontsize=11)
    ax.set_title("A. Per-Layer Correlation with Human Regression-In", fontsize=12, fontweight='bold')
    ax.set_xticks(range(num_layers))
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add brain region annotations
    for region_name, layers in brain_map.items():
        short = brain_labels_short[region_name]
        mid = (layers[0] + layers[-1]) / 2
        ax.annotate(short, xy=(mid, ax.get_ylim()[1] * 0.95),
                    ha='center', fontsize=8, fontstyle='italic',
                    color='darkblue', alpha=0.7)

    # ── Panel B: Per-layer bar chart (Reading Time) ──
    ax = axes[0, 1]
    ax.bar(layer_df["Layer"], layer_df["r_reading_time"],
           color=colors_by_layer, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("GPT-2 Layer", fontsize=11)
    ax.set_ylabel("Pearson r (vs Reading Time)", fontsize=11)
    ax.set_title("B. Per-Layer Correlation with Reading Time", fontsize=12, fontweight='bold')
    ax.set_xticks(range(num_layers))
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # ── Panel C: Brain region grouped comparison ──
    ax = axes[1, 0]
    if not region_df.empty:
        x = np.arange(len(region_df))
        w = 0.3
        ax.bar(x - w, region_df["r_regression_in"], w, label="vs Regression-In",
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(x, region_df["r_regression_out"], w, label="vs Regression-Out",
               color='#f39c12', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(x + w, region_df["r_reading_time"], w, label="vs Reading Time",
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(region_df["Region"], fontsize=8, ha='center')
        ax.set_ylabel("Pearson r", fontsize=11)
        ax.set_title("C. Brain Region Mapping: Correlation Summary", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # ── Panel D: Schematic brain diagram ──
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("D. Tentative Layer-to-Brain Mapping", fontsize=12, fontweight='bold')

    # Draw a simplified brain outline
    theta = np.linspace(0, 2 * np.pi, 100)
    brain_x = 5 + 3.5 * np.cos(theta) * (1 + 0.15 * np.cos(2 * theta))
    brain_y = 5 + 3.0 * np.sin(theta) * (1 + 0.1 * np.cos(3 * theta))
    ax.fill(brain_x, brain_y, color='#f5f5f5', edgecolor='#333', linewidth=2)

    # Brain regions
    regions_draw = [
        # (x, y, label, color, brain_region)
        (3.0, 6.5, "Layers 0–3\nVisual/Ortho", '#3498db', "V1 / VWFA"),
        (5.0, 4.0, "Layers 4–7\nSyntactic", '#e74c3c', "Broca's / L-IFG"),
        (7.0, 6.5, "Layers 8–11\nSemantic", '#2ecc71', "Wernicke's / AG"),
    ]

    for x, y, label, color, brain_label in regions_draw:
        circle = plt.Circle((x, y), 1.2, color=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y + 0.2, label, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, y - 0.7, brain_label, ha='center', va='center', fontsize=7,
                fontstyle='italic', color='#555')

        # Add correlation value
        if not region_df.empty:
            match = region_df[region_df["Region_short"].str.contains(
                label.split("\n")[1][:5], case=False)]
            if not match.empty:
                r_val = match.iloc[0]["r_regression_in"]
                ax.text(x, y - 1.1, f"r = {r_val:+.3f}", ha='center', fontsize=8,
                        color=color, fontweight='bold')

    # Arrows showing information flow
    ax.annotate("", xy=(4.3, 4.5), xytext=(3.2, 5.5),
                arrowprops=dict(arrowstyle="->", color='gray', lw=1.5))
    ax.annotate("", xy=(5.7, 4.5), xytext=(6.8, 5.5),
                arrowprops=dict(arrowstyle="->", color='gray', lw=1.5))

    plt.suptitle(f"Layer-to-Brain Mapping Analysis — GPT-2 × Dundee",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/layer_brain_mapping_{model_name}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Print results ──
    print(f"\n  Layer-to-Brain Mapping ({model_name}):")
    print("  " + "-" * 70)
    print(f"  {'Layer':8s} {'r(RegIn)':>10s} {'p-value':>10s} {'r(RT)':>10s} {'Brain Region':>20s}")
    print("  " + "-" * 70)
    for _, row in layer_df.iterrows():
        l = int(row["Layer"])
        if l < 4:
            region = "Visual/Ortho (V1)"
        elif l < 8:
            region = "Syntactic (Broca's)"
        else:
            region = "Semantic (Wernicke's)"
        sig = "*" if row["p_regression_in"] < 0.05 else ""
        print(f"  L{l:2d}    {row['r_regression_in']:+.4f}    {row['p_regression_in']:.2e}  "
              f"{row['r_reading_time']:+.4f}    {region} {sig}")
    print("  " + "-" * 70)
    print(f"  Saved → {output_dir}/layer_brain_mapping_{model_name}.png\n")

    return layer_df, region_df


# ============================================================
# 6. Reading Time Effects
# ============================================================

def plot_reading_time_effects(df, model_name, output_dir="results/images"):
    """Scatter plots of reading time vs entropy and surprisal."""
    os.makedirs(output_dir, exist_ok=True)

    if "reading_time" not in df.columns or df["reading_time"].sum() == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reading Time vs Entropy
    sns.regplot(data=df[df["reading_time"] > 0], x="reading_time", y="entropy_mean",
                scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'},
                ax=axes[0])
    axes[0].set_title(f"Reading Time vs Attention Entropy ({model_name})", fontweight='bold')

    # Reading Time vs Surprisal
    sns.regplot(data=df[df["reading_time"] > 0], x="reading_time", y="surprisal",
                scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'},
                ax=axes[1])
    axes[1].set_title(f"Reading Time vs Surprisal ({model_name})", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/reading_time_effects_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved reading time plot → {output_dir}/reading_time_effects_{model_name}.png")


# ============================================================
# 7. Additional Scatter Plots
# ============================================================

def plot_surprisal_vs_entropy(df, model_name, output_dir="results/images"):
    """Surprisal vs Mean Attention Entropy (regression-entropy relationship)."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Surprisal vs Entropy
    sns.regplot(data=df, x="surprisal", y="entropy_mean",
                scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'}, ax=axes[0])
    axes[0].set_title("Surprisal vs Mean Attention Entropy", fontweight='bold')
    axes[0].set_xlabel("Surprisal S(w)")
    axes[0].set_ylabel("Mean Attention Entropy H(A)")

    # Surprisal vs Regression-In
    sns.regplot(data=df, x="surprisal", y="regression_in",
                scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'blue'}, ax=axes[1])
    axes[1].set_title("Surprisal vs Regression-In Probability", fontweight='bold')
    axes[1].set_xlabel("Surprisal S(w)")
    axes[1].set_ylabel("Regression-In P")

    plt.suptitle(f"LLM Uncertainty vs Human Metrics ({model_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/surprisal_entropy_scatter_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter plot → {output_dir}/surprisal_entropy_scatter_{model_name}.png")


# ============================================================
# 8. Comprehensive Text Report
# ============================================================

def save_comprehensive_report(df, corrs, head_corrs, model_name, dataset_name,
                               output_dir="results/texts"):
    """Generate a detailed text report of all findings."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/comprehensive_report_{model_name}_{dataset_name}.txt"

    lines = []
    lines.append("=" * 72)
    lines.append(f"COGNITIVE SYMMETRY: COMPREHENSIVE REPORT")
    lines.append(f"Model: {model_name}  |  Dataset: {dataset_name}")
    lines.append("=" * 72)
    lines.append("")

    # Dataset stats
    lines.append(f"Data Summary:")
    lines.append(f"  Total aligned word-tokens:  {len(df)}")
    lines.append(f"  Total unique sentences:     {df['sentence'].nunique()}")
    if "text_id" in df.columns:
        lines.append(f"  Text files processed:       {df['text_id'].nunique()}")
    lines.append(f"  Surprisal range:            [{df['surprisal'].min():.2f}, {df['surprisal'].max():.2f}]")
    lines.append(f"  Mean entropy:               {df['entropy_mean'].mean():.4f} ± {df['entropy_mean'].std():.4f}")
    lines.append(f"  Mean regression-in prob:     {df['regression_in'].mean():.4f}")
    lines.append(f"  Mean regression-out prob:    {df['regression_out'].mean():.4f}")
    if "reading_time" in df.columns:
        rt = df[df["reading_time"] > 0]["reading_time"]
        lines.append(f"  Mean reading time:           {rt.mean():.1f} ms (σ = {rt.std():.1f})")
    lines.append("")

    # Correlations
    lines.append("-" * 50)
    lines.append("CORRELATION RESULTS (Pearson r)")
    lines.append("-" * 50)
    for name, vals in corrs.items():
        sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else "ns"
        lines.append(f"  {name:30s}: r = {vals['r']:+.4f}  p = {vals['p']:.2e}  {sig}")
    lines.append("")

    # Top saccadic heads
    if head_corrs:
        lines.append("-" * 50)
        lines.append("TOP SACCADIC HEADS (by |r| with Regression-In)")
        lines.append("-" * 50)
        sorted_heads = sorted(head_corrs, key=lambda x: abs(x["Correlation"]), reverse=True)[:10]
        for h in sorted_heads:
            lines.append(f"  L{h['Layer']:2d} H{h['Head']:2d}: r = {h['Correlation']:+.4f}  p = {h['p_value']:.2e}")
        lines.append("")

    # Sentence structure
    if "sentence_structure" in df.columns:
        lines.append("-" * 50)
        lines.append("SENTENCE STRUCTURE BREAKDOWN")
        lines.append("-" * 50)
        for struct in sorted(df["sentence_structure"].unique()):
            sub = df[df["sentence_structure"] == struct]
            lines.append(f"  {struct}: {sub['sentence'].nunique()} sentences, {len(sub)} words")
        lines.append("")

    # Interpretation
    lines.append("-" * 50)
    lines.append("INTERPRETATION")
    lines.append("-" * 50)

    surp_r = corrs.get("surprisal_vs_reg_in", {}).get("r", 0)
    ent_r = corrs.get("entropy_vs_reg_in", {}).get("r", 0)

    if surp_r > 0.15:
        lines.append(f"  ✓ POSITIVE surprisal–regression correlation (r={surp_r:.3f}).")
        lines.append(f"    → Validates the Expectation-Based theory: words unexpected to GPT-2")
        lines.append(f"      trigger real human back-tracking behaviour.")
    elif surp_r > 0:
        lines.append(f"  ~ Weak positive surprisal–regression correlation (r={surp_r:.3f}).")
    else:
        lines.append(f"  ✗ No positive surprisal–regression correlation (r={surp_r:.3f}).")

    lines.append("")
    if ent_r > 0.1:
        lines.append(f"  ✓ POSITIVE entropy–regression correlation (r={ent_r:.3f}).")
        lines.append(f"    → When humans look back, GPT-2's attention also 'searches' broadly.")
        lines.append(f"      Supports the Cognitive Symmetry hypothesis.")
    elif ent_r > 0:
        lines.append(f"  ~ Weak positive entropy–regression correlation (r={ent_r:.3f}).")
    else:
        lines.append(f"  ✗ No positive entropy–regression correlation (r={ent_r:.3f}).")

    lines.append("")
    lines.append("=" * 72)

    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved comprehensive report → {report_path}")
