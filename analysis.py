import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

def align_and_correlate(human_data, llm_data):
    """
    human_data: list of dicts {"sentence", "metrics": [{"word", "regression_in_prob"}]}
    llm_data: list of dicts {"sentence", "metrics": [{"word", "surprisal", "attention_entropy": {}}]}
    """
    results = []
    
    # We flatten across all sentences
    for h_sent, l_sent in zip(human_data, llm_data):
        h_metrics = h_sent["metrics"]
        l_metrics = l_sent["metrics"]
        
        for h_w, l_w in zip(h_metrics, l_metrics):
            if h_w["word"] == l_w["word"]:
                row = {
                    "word": h_w["word"],
                    "regression_in": h_w["regression_in_prob"],
                    "regression_out": h_w["regression_out_prob"],
                    "reading_time": h_w.get("reading_time", 0),
                    "surprisal": l_w["surprisal"]
                }
                
                # Add all layer entropies
                for k, v in l_w["attention_entropy"].items():
                    row[f"entropy_{k}"] = v
                    
                # overall attention entropy (mean across all heads/layers)
                if len(l_w["attention_entropy"]) > 0:
                    row["entropy_mean"] = np.mean(list(l_w["attention_entropy"].values()))
                else:
                    row["entropy_mean"] = 0
                    
                results.append(row)
                
    df = pd.DataFrame(results)
    
    # Compute overall correlations
    if len(df) > 1:
        corr_ent_in, _ = pearsonr(df["entropy_mean"], df["regression_in"])
        corr_ent_out, _ = pearsonr(df["entropy_mean"], df["regression_out"])
        corr_surp_in, _ = pearsonr(df["surprisal"], df["regression_in"])
        corr_rt_ent, _ = pearsonr(df["reading_time"], df["entropy_mean"]) if "reading_time" in df else (0, 0)
        corr_rt_surp, _ = pearsonr(df["reading_time"], df["surprisal"]) if "reading_time" in df else (0, 0)
    else:
        corr_ent_in = corr_ent_out = corr_surp_in = corr_rt_ent = corr_rt_surp = 0
        
    print(f"Overall Pearson r (Entropy Mean vs Regression-In): {corr_ent_in:.3f}")
    print(f"Overall Pearson r (Entropy Mean vs Regression-Out): {corr_ent_out:.3f}")
    print(f"Overall Pearson r (Surprisal vs Regression-In): {corr_surp_in:.3f}")
    if "reading_time" in df:
        print(f"Overall Pearson r (Reading Time vs Entropy Mean): {corr_rt_ent:.3f}")
        print(f"Overall Pearson r (Reading Time vs Surprisal): {corr_rt_surp:.3f}")
    
    return df, {
        "corr_ent_in": corr_ent_in, 
        "corr_ent_out": corr_ent_out, 
        "corr_surp_in": corr_surp_in,
        "corr_rt_ent": corr_rt_ent,
        "corr_rt_surp": corr_rt_surp
    }

def save_text_results(corrs, model_name, dataset, output_dir="results/texts"):
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/summary_{model_name}_{dataset}.txt"
    with open(report_path, "w") as f:
        f.write(f"=== Cognitive Symmetry Report: {model_name} on {dataset} ===\n\n")
        f.write("Metrics Explanation:\n")
        f.write("- Regression-In Rate: How often humans 'look back' to this word during reading.\n")
        f.write("- Mean Attention Entropy: The average spread of the LLM's attention heads. High entropy means the model is 'searching' its context window broadly.\n")
        f.write("- Surprisal: Information theory measure of how unexpected the word is to the LLM.\n\n")
        
        f.write("Results:\n")
        f.write(f"1. Correlation (Attention Entropy vs Human Regression-In): {corrs['corr_ent_in']:.4f}\n")
        f.write("   -> If positive and significant (e.g., > 0.3), this suggests that when human readers struggle to integrate a word and look back, the LLM also struggles and 'looks broadly' across its context via high attention entropy.\n\n")
        
        f.write(f"2. Correlation (Word Surprisal vs Human Regression-In): {corrs['corr_surp_in']:.4f}\n")
        f.write("   -> If positive, it validates the Expectation-Based Syntactic Comprehension theory: words highly unexpected (surprising) to the LLM cause humans to backtrack.\n\n")
        
        f.write(f"3. Correlation (Attention Entropy vs Human Regression-Out): {corrs['corr_ent_out']:.4f}\n")
        f.write("   -> Indicates if the LLM's attention spreads precisely where humans *originate* their backwards look.\n")

    print(f"Saved textual results to {report_path}")

def plot_saccadic_heads(df, model_name, output_dir="results/images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # We want to find heads whose entropy correlates highly with human regressions
    entropy_cols = [c for c in df.columns if c.startswith("entropy_L")]
    
    head_corrs = []
    for col in entropy_cols:
        r, _ = pearsonr(df[col], df["regression_in"])
        # extract L and H
        # format: entropy_L0_H1
        parts = col.split("_")
        layer = int(parts[1][1:])
        head = int(parts[2][1:])
        head_corrs.append({"Layer": layer, "Head": head, "Correlation": r})
        
    head_df = pd.DataFrame(head_corrs)
    if not head_df.empty:
        pivot_df = head_df.pivot(index="Layer", columns="Head", values="Correlation")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, cmap="coolwarm", center=0, annot=False)
        plt.title(f"Saccadic Heads Localization ({model_name}) - Correlation with Regression-In")
        plt.savefig(f"{output_dir}/saccadic_heads_{model_name}.png")
        plt.close()
        print(f"Saved Saccadic Heads heat map to {output_dir}/saccadic_heads_{model_name}.png")

def plot_reading_time_effects(df, model_name, output_dir="results/images"):
    os.makedirs(output_dir, exist_ok=True)
    if "reading_time" not in df.columns or df["reading_time"].sum() == 0:
        return
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.regplot(data=df, x="reading_time", y="entropy_mean", scatter_kws={'alpha':0.5})
    plt.title(f"Reading Time vs Attn Entropy ({model_name})")
    
    plt.subplot(1, 2, 2)
    sns.regplot(data=df, x="reading_time", y="surprisal", scatter_kws={'alpha':0.5})
    plt.title(f"Reading Time vs Surprisal ({model_name})")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reading_time_effects_{model_name}.png")
    plt.close()
    print(f"Saved Reading Time effects plot to {output_dir}/reading_time_effects_{model_name}.png")

def plot_regression_entropy_lag(df, model_name, output_dir="results/images"):
    # Group by normalized surprisal bins
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x="surprisal", y="entropy_mean", scatter_kws={'alpha':0.5})
    plt.title(f"Regression-Entropy Lag: Surprisal vs Average Attention Entropy ({model_name})")
    plt.xlabel("Surprisal")
    plt.ylabel("Mean Attention Entropy")
    plt.savefig(f"{output_dir}/surprisal_vs_entropy_{model_name}.png")
    plt.close()
    print(f"Saved Regression-Entropy Lag plot to {output_dir}/surprisal_vs_entropy_{model_name}.png")

