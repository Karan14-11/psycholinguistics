# Cognitive Symmetry: LLM Attention vs. Human Saccades

This project mathematically correlates human eye-tracking behavior—specifically "look-backs" or regressive saccades during complex parsing—against the internal processing states of Large Language Models (LLMs).

## Overview

Human language processing is inherently non-linear; readers frequently pause and backtrack when encountering linguistic ambiguity (e.g., "garden-path" sentences). 

This codebase evaluates multiple architectures (`GPT-2` and `BERT`) by comparing them to classical eye-tracking corpora (Dundee and GECO). Specifically, it tracks:
- **LLM Surprisal ($S$)**: Cross-entropy tracking how unexpected a word is given prior context.
- **Attention Entropy ($H$)**: A Shannon entropy measurement that reveals whether a specific attention head is highly focused on a single token, or 'searching' a broad span of previous tokens (mimicking a regression).

We aim to discover **Saccadic Heads** (layers/heads that specialize in syntax checking) and analyze the **Regression-Entropy Lag**.

## Code Architecture

* **`data_utils.py`**: Parses GECO and Dundee eye-tracking metrics (e.g., probability of regression-in and out). If local data files aren't found in a `data/` directory, it automatically mocks synthetic "garden-path" data to test the pipeline end-to-end.
* **`metrics.py`**: Houses the raw mathematical implementations for Surprisal ($S(w_i) = -\log P(w_i|w_{<i})$) and Attention Entropy ($H(A_i) = -\sum A_{ij} \log A_{ij}$). Includes special handling for Masked LMs like BERT.
* **`llm_pipeline.py`**: Coordinates the Hugging Face `transformers` forward passes, capturing hidden states/attentions and mapping subword token measurements back into coherent word-level alignments. 
* **`analysis.py`**: The statistical engine computing Pearson $r$ correlations between human metrics and model uncertainty, rendering outputs via `matplotlib`/`seaborn`.
* **`main.py`**: The overarching script that pulls from utils, runs the models, performs the analysis, and saves output plots.

## Setup Instructions

1. **Install Dependencies**
   Ensure you have PyTorch and Transformers installed exactly as prescribed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Configuration**
   By default, the script will execute using automatically generated mock logic.
   * To use real eye-tracking metrics, create a directory named `data/` at the root of the project.
   * Add your data files into this directory labeled as `data/dundee.csv` or `data/geco.csv`.

3. **Execution**
   Launch the main pipeline.
   ```bash
   python main.py
   ```
   *Generated output graphics (heatmaps and scatterplots) will be cleanly exported to a `results/` directory.*
