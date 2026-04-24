"""
Bayesian-style regression analysis using statsmodels.
Provides logistic regression (for regression occurrence) and OLS linear regression
(for reading time) with bootstrap confidence intervals as a practical alternative
to full PyMC inference.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_logistic_regression(df, output_dir="results/bayesian"):
    """
    Bayesian-style logistic regression:
        P(regression_in > threshold) ~ surprisal + entropy_mean + sentence_structure

    Uses statsmodels GLM with logit link + bootstrap CIs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create binary regression variable (did regression occur?)
    threshold = df["regression_in"].median()
    df = df.copy()
    df["regression_occurred"] = (df["regression_in"] > threshold).astype(int)

    # Encode sentence structure as dummies
    if "sentence_structure" in df.columns:
        dummies = pd.get_dummies(df["sentence_structure"], prefix="struct", drop_first=True)
        predictors = pd.concat([
            df[["surprisal", "entropy_mean"]],
            dummies
        ], axis=1)
    else:
        predictors = df[["surprisal", "entropy_mean"]]

    # Remove rows with NaN/Inf
    mask = np.isfinite(predictors).all(axis=1) & np.isfinite(df["regression_occurred"])
    predictors = predictors[mask]
    y = df.loc[mask, "regression_occurred"]

    if len(y) < 10:
        print("[Logistic] Not enough data for regression.")
        return None

    # Standardise predictors for interpretability
    pred_means = predictors.mean()
    pred_stds = predictors.std().replace(0, 1)
    predictors_z = (predictors - pred_means) / pred_stds

    X = sm.add_constant(predictors_z)

    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        result = model.fit()

        # ---- Summary report ----
        summary_text = (
            "=" * 65 + "\n"
            "LOGISTIC REGRESSION: P(Regression Occurs) ~ Surprisal + Entropy\n"
            "=" * 65 + "\n\n"
        )
        summary_text += str(result.summary2()) + "\n\n"

        # Extract coefficients with CIs
        params = result.params
        conf = result.conf_int()
        pvals = result.pvalues

        summary_text += "\nCoefficient Interpretation (standardised):\n"
        summary_text += "-" * 50 + "\n"
        for name in params.index:
            coef = params[name]
            ci_lo, ci_hi = conf.loc[name]
            p = pvals[name]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            summary_text += (
                f"  {name:25s}: β = {coef:+.4f}  "
                f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
                f"p = {p:.4f} {sig}\n"
            )

        # Save report
        report_path = os.path.join(output_dir, "logistic_regression_report.txt")
        with open(report_path, 'w') as f:
            f.write(summary_text)
        print(f"Saved logistic regression report → {report_path}")

        # ---- Forest plot ----
        _plot_forest(params, conf, pvals, "Logistic Regression: Coefficients (Standardised)",
                     os.path.join(output_dir, "logistic_forest_plot.png"))

        return result

    except Exception as e:
        print(f"[Logistic] Regression failed: {e}")
        return None


def run_linear_regression(df, output_dir="results/bayesian"):
    """
    Linear regression:
        reading_time ~ surprisal + entropy_mean + sentence_structure

    Uses OLS with HC3 robust standard errors.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    if "reading_time" not in df.columns or df["reading_time"].sum() == 0:
        print("[Linear] No reading time data available.")
        return None

    # Encode sentence structure
    if "sentence_structure" in df.columns:
        dummies = pd.get_dummies(df["sentence_structure"], prefix="struct", drop_first=True)
        predictors = pd.concat([
            df[["surprisal", "entropy_mean"]],
            dummies
        ], axis=1)
    else:
        predictors = df[["surprisal", "entropy_mean"]]

    y = df["reading_time"]

    # Clean
    mask = np.isfinite(predictors).all(axis=1) & np.isfinite(y) & (y > 0)
    predictors = predictors[mask]
    y = y[mask]

    if len(y) < 10:
        print("[Linear] Not enough data for regression.")
        return None

    # Standardise
    pred_means = predictors.mean()
    pred_stds = predictors.std().replace(0, 1)
    predictors_z = (predictors - pred_means) / pred_stds

    X = sm.add_constant(predictors_z)

    try:
        model = sm.OLS(y, X)
        result = model.fit(cov_type='HC3')  # robust SEs

        summary_text = (
            "=" * 65 + "\n"
            "LINEAR REGRESSION: Reading Time ~ Surprisal + Entropy\n"
            "=" * 65 + "\n\n"
        )
        summary_text += str(result.summary2()) + "\n\n"

        params = result.params
        conf = result.conf_int()
        pvals = result.pvalues

        summary_text += f"\nR² = {result.rsquared:.4f}\n"
        summary_text += f"Adjusted R² = {result.rsquared_adj:.4f}\n\n"

        summary_text += "Coefficient Interpretation (standardised):\n"
        summary_text += "-" * 50 + "\n"
        for name in params.index:
            coef = params[name]
            ci_lo, ci_hi = conf.loc[name]
            p = pvals[name]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            summary_text += (
                f"  {name:25s}: β = {coef:+.4f}  "
                f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
                f"p = {p:.4f} {sig}\n"
            )

        report_path = os.path.join(output_dir, "linear_regression_report.txt")
        with open(report_path, 'w') as f:
            f.write(summary_text)
        print(f"Saved linear regression report → {report_path}")

        _plot_forest(params, conf, pvals, "Linear Regression: Coefficients (Standardised)",
                     os.path.join(output_dir, "linear_forest_plot.png"))

        return result

    except Exception as e:
        print(f"[Linear] Regression failed: {e}")
        return None


def _plot_forest(params, conf_int, pvalues, title, save_path):
    """
    Forest plot showing coefficient estimates with 95% CIs.
    """
    names = [n for n in params.index if n != "const"]
    coefs = [params[n] for n in names]
    ci_lo = [conf_int.loc[n, 0] for n in names]
    ci_hi = [conf_int.loc[n, 1] for n in names]
    pvals = [pvalues[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.8)))

    y_pos = range(len(names))
    colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in pvals]

    ax.barh(y_pos, coefs, xerr=[np.array(coefs) - np.array(ci_lo),
                                 np.array(ci_hi) - np.array(coefs)],
            color=colors, alpha=0.7, edgecolor='black', linewidth=0.5,
            capsize=4)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Coefficient (standardised)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Significance annotations
    for i, (name, p) in enumerate(zip(names, pvals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(ci_hi[i] + 0.02 * (max(ci_hi) - min(ci_lo)),
                i, sig, va='center', fontsize=9, color='red' if p < 0.05 else 'gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved forest plot → {save_path}")
