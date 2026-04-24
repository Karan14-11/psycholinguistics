"""
Generates a professional PDF report for the Cognitive Symmetry mid-submission.
Uses fpdf2 for layout with embedded result images.
"""

import os
from fpdf import FPDF


class CognitiveSymmetryReport(FPDF):

    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(auto=True, margin=20)

    # ── Header / Footer ──
    def header(self):
        if self.page_no() == 1:
            return  # title page has its own header
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, "Cognitive Symmetry: Mid-Submission Report", align="L")
        self.cell(0, 5, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 12, 200, 12)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── Helpers ──
    def section_title(self, title, level=1):
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(30, 30, 120)
            self.ln(4)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(30, 30, 120)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)
        elif level == 2:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(60, 60, 60)
            self.ln(3)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)
        elif level == 3:
            self.set_font("Helvetica", "BI", 11)
            self.set_text_color(80, 80, 80)
            self.ln(2)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=15):
        x = self.get_x()
        self.set_x(x + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        w = self.w - self.r_margin - self.get_x()
        self.multi_cell(w, 5.5, f"-  {text}")
        self.ln(0.5)

    def add_image_centered(self, path, w=170, caption=""):
        if not os.path.exists(path):
            self.body_text(f"[Image not found: {path}]")
            return
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        if caption:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(40, 40, 100)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(235, 235, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(2)

    def finding_box(self, text, color="blue"):
        """Highlighted finding box."""
        colors = {
            "blue": (220, 230, 245),
            "green": (220, 245, 220),
            "orange": (255, 240, 220),
        }
        bg = colors.get(color, colors["blue"])
        self.set_fill_color(*bg)
        self.set_draw_color(100, 100, 160)
        x = self.get_x()
        y = self.get_y()
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(30, 30, 100)

        # Calculate height
        self.set_x(x + 5)
        w = 180
        line_height = 5
        # Use multi_cell to auto-wrap
        self.rect(x, y, 190, self._estimate_box_height(text, w, line_height), style="DF")
        self.set_xy(x + 5, y + 2)
        self.multi_cell(w, line_height, text)
        self.ln(2)

    def _estimate_box_height(self, text, w, lh):
        lines = len(text) / (w / 2.2) + 1  # rough estimate
        return max(lines * lh + 6, 12)


def build_report():
    pdf = CognitiveSymmetryReport()
    pdf.alias_nb_pages()

    img_dir = "results/images"
    bayes_dir = "results/bayesian"

    # ================================================================
    # TITLE PAGE
    # ================================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(30, 30, 120)
    pdf.cell(0, 14, "Cognitive Symmetry", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Modeling Human Regressive Saccades via", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "LLM Attention Entropy and Surprisal", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_draw_color(30, 30, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Karan Nijhawan (2022101122)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Nidhi Vaidya (2023114005)", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Computational Psycholinguistics - Mid Submission Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "April 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, "Model: GPT-2 (Low-Tier)  |  Corpus: Dundee Eye-Tracking Corpus", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "4,585 aligned word-tokens  |  200 sentences  |  10 subjects", align="C", new_x="LMARGIN", new_y="NEXT")

    # ================================================================
    # ABSTRACT
    # ================================================================
    pdf.add_page()
    pdf.section_title("Abstract")
    pdf.body_text(
        "This report presents the mid-submission results of our Cognitive Symmetry study, which investigates "
        "the correspondence between human eye-tracking behavior (specifically regressive saccades) and the "
        "internal processing states of large language models. Using GPT-2 as our low-tier model and the Dundee "
        "Eye-Tracking Corpus (3 texts, 10 subjects, 4,585 aligned word-tokens), we compute token-level surprisal "
        "and layer-wise attention entropy from the transformer, then correlate these with word-level regression-in "
        "probabilities and reading times measured from human subjects."
    )
    pdf.body_text(
        "Our results confirm all core hypotheses from the project proposal. Both surprisal (r = +0.173, "
        "p < 10^-32) and attention entropy (r = +0.131, p < 10^-19) significantly predict human regressive "
        "saccades. We identify specific 'saccadic heads' - attention heads that mirror human back-tracking "
        "behavior - with the strongest being Layer 1, Head 11 (r = 0.434). Our novel layer-to-brain mapping "
        "analysis reveals that middle transformer layers (4-7), hypothesized to correspond to Broca's area, "
        "show the strongest alignment with human regression behavior. We also demonstrate a regression-entropy "
        "lag: GPT-2's attention broadens 1-2 tokens after the high-surprisal word, while humans react "
        "immediately at that word."
    )

    # ================================================================
    # 1. INTRODUCTION
    # ================================================================
    pdf.section_title("1. Introduction")
    pdf.body_text(
        "Human language processing is non-linear. When readers encounter syntactically ambiguous sentences "
        "(e.g., garden-path sentences like 'The old man the boat'), they experience a breakdown in their "
        "initial parse, leading to regressive saccades - physical backward eye movements to re-read earlier "
        "words. These regressions occur approximately 10-15% of the time during normal reading and spike "
        "significantly in syntactically complex constructions."
    )
    pdf.body_text(
        "This study extends the framework proposed by Futrell et al. (2019), which suggests that LLMs can "
        "serve as 'psycholinguistic subjects,' by analyzing whether LLM uncertainty - quantified through "
        "Information Theory metrics - aligns with physical eye-movement data from the Dundee Eye-Tracking Corpus."
    )
    pdf.body_text(
        "For this mid-submission, we focus on a single model-dataset combination (GPT-2 x Dundee) to "
        "establish the core methodology and validate the primary hypotheses before expanding to the full "
        "2x2 design (GPT-2/LLaMA 3 x Dundee/GECO) in the final submission."
    )

    # ================================================================
    # 2. METHODOLOGY
    # ================================================================
    pdf.section_title("2. Methodology")

    pdf.section_title("2.1 Metrics of Uncertainty", level=2)
    pdf.bold_text("LLM Surprisal: S(w_i) = -log P(w_i | w_<i)")
    pdf.body_text(
        "Surprisal measures how unexpected a word is given its preceding context. Computed via cross-entropy "
        "loss from GPT-2's causal language modeling head. Higher surprisal indicates the model did not predict "
        "the word well, suggesting processing difficulty."
    )
    pdf.bold_text("Attention Entropy: H(A_i) = - sum_j A_ij log A_ij")
    pdf.body_text(
        "Shannon entropy of the attention distribution for each head at each token position. High entropy "
        "indicates the model is distributing attention broadly across the context ('searching'), while low "
        "entropy indicates focused attention on specific tokens."
    )
    pdf.bold_text("Human Metrics: Regression-In/Out Probability, Reading Time")
    pdf.body_text(
        "Regression-In probability measures the proportion of subjects who made a backward saccade to a given "
        "word. Regression-Out measures the proportion who saccaded backward from that word. Reading time is "
        "the average first-fixation duration across subjects."
    )

    pdf.section_title("2.2 Statistical Framework", level=2)
    pdf.body_text(
        "We employ three complementary statistical approaches:"
    )
    pdf.bullet("Pearson correlation analysis between LLM metrics and human eye-tracking measures")
    pdf.bullet("Logistic regression: P(regression occurs) ~ surprisal + entropy + sentence_structure")
    pdf.bullet("OLS linear regression: reading_time ~ surprisal + entropy + sentence_structure")
    pdf.body_text(
        "Logistic and linear regressions use standardized predictors with robust (HC3) standard errors."
    )

    pdf.section_title("2.3 Dataset", level=2)
    pdf.add_table(
        ["Parameter", "Value"],
        [
            ["Corpus", "Dundee Eye-Tracking Corpus"],
            ["Texts Processed", "3 (of 20 available)"],
            ["Subjects", "10 (sa through sj)"],
            ["Sentences", "200"],
            ["Aligned Word-Tokens", "4,585"],
            ["Sentence Structures", "146 active, 38 passive, 16 relative clause"],
        ],
        col_widths=[60, 130]
    )

    # ================================================================
    # 3. RESULTS
    # ================================================================
    pdf.section_title("3. Results")

    # 3.1 Core Correlations
    pdf.section_title("3.1 Core Correlations", level=2)
    pdf.body_text(
        "All primary hypotheses from the proposal are confirmed with extremely high statistical significance:"
    )
    pdf.add_table(
        ["Metric Pair", "Pearson r", "p-value", "Significance"],
        [
            ["Surprisal vs Regression-In", "+0.173", "4.0 x 10^-32", "***"],
            ["Entropy vs Regression-In", "+0.131", "5.0 x 10^-19", "***"],
            ["Surprisal vs Regression-Out", "+0.059", "6.3 x 10^-5", "***"],
            ["Entropy vs Regression-Out", "+0.040", "6.7 x 10^-3", "**"],
            ["Reading Time vs Surprisal", "+0.216", "2.5 x 10^-49", "***"],
            ["Reading Time vs Entropy", "+0.141", "1.2 x 10^-21", "***"],
        ],
        col_widths=[60, 30, 50, 50]
    )
    pdf.body_text(
        "The strongest correlation is between reading time and surprisal (r = +0.216), indicating that "
        "words GPT-2 finds unexpected also take humans longer to fixate on. Surprisal consistently "
        "outperforms entropy as a predictor, consistent with the hypothesis that prediction error is the "
        "primary driver of processing difficulty."
    )

    # Scatter plot
    scatter_path = os.path.join(img_dir, "surprisal_entropy_scatter_GPT2_DUNDEE.png")
    pdf.add_image_centered(scatter_path, w=175,
                           caption="Figure 1: Left - Surprisal vs Mean Attention Entropy; Right - Surprisal vs Regression-In Probability")

    # 3.2 Saccadic Heads
    pdf.add_page()
    pdf.section_title("3.2 Saccadic Heads Identification", level=2)
    pdf.body_text(
        "A key novel contribution is the identification of specific attention heads whose entropy correlates "
        "strongly with human regression behavior. We term these 'saccadic heads' - transformer components "
        "that functionally mirror the human 'look-back' mechanism."
    )

    saccadic_path = os.path.join(img_dir, "saccadic_heads_GPT2_DUNDEE.png")
    pdf.add_image_centered(saccadic_path, w=165,
                           caption="Figure 2: Saccadic Heads Heatmap - Per-head Pearson r with human regression-in probability")

    pdf.body_text(
        "The heatmap reveals that head-level correlations vary dramatically (from -0.12 to +0.43), indicating "
        "that specific heads specialize in processing that mirrors human syntactic reanalysis."
    )

    pdf.section_title("Top 10 Saccadic Heads", level=3)
    pdf.add_table(
        ["Rank", "Layer", "Head", "Pearson r", "p-value"],
        [
            ["1", "1", "11", "+0.434", "1.5 x 10^-210"],
            ["2", "7", "4", "+0.334", "4.5 x 10^-120"],
            ["3", "10", "5", "+0.314", "2.2 x 10^-105"],
            ["4", "9", "4", "+0.286", "1.0 x 10^-86"],
            ["5", "11", "5", "+0.239", "1.3 x 10^-60"],
            ["6", "6", "3", "+0.232", "3.5 x 10^-57"],
            ["7", "9", "9", "+0.204", "3.2 x 10^-44"],
            ["8", "4", "8", "+0.203", "8.1 x 10^-44"],
            ["9", "0", "5", "+0.202", "2.5 x 10^-43"],
            ["10", "11", "7", "+0.201", "5.1 x 10^-43"],
        ],
        col_widths=[20, 25, 25, 40, 80]
    )

    pdf.body_text(
        "Head L1-H11 stands out as exceptionally strong (r = 0.434), far exceeding the overall mean "
        "entropy correlation (r = 0.131). Its position in an early layer (Layer 1) suggests that GPT-2 "
        "detects reading difficulty very early in processing. The top heads span early (L0-L1), middle "
        "(L4-L7), and late (L9-L11) layers, indicating distributed processing."
    )

    # 3.3 Layer-to-Brain Mapping
    pdf.add_page()
    pdf.section_title("3.3 Layer-to-Brain Mapping", level=2)
    pdf.body_text(
        "We introduce a tentative mapping from GPT-2's 12 transformer layers to known brain regions "
        "involved in reading and language comprehension:"
    )
    pdf.bullet("Early layers (L0-L3): Visual/Orthographic processing (V1, Visual Word Form Area)")
    pdf.bullet("Middle layers (L4-L7): Syntactic parsing (Broca's area, Left Inferior Frontal Gyrus)")
    pdf.bullet("Late layers (L8-L11): Semantic integration (Wernicke's area, Angular Gyrus)")
    pdf.ln(2)

    brain_path = os.path.join(img_dir, "layer_brain_mapping_GPT2_DUNDEE.png")
    pdf.add_image_centered(brain_path, w=175,
                           caption="Figure 3: Layer-to-Brain Mapping Analysis (4 panels)")

    pdf.body_text(
        "Panel A shows per-layer correlation with regression-in. Middle layers (4-7) consistently achieve "
        "the highest correlations (~0.15-0.17), supporting the hypothesis that syntactic processing stages "
        "in the transformer most closely mirror human reanalysis behavior. Panel C confirms this at the "
        "grouped brain-region level: the 'Broca's area' tier (middle layers) achieves r = 0.156, followed "
        "by early (r = 0.117) and late (r = 0.116) tiers."
    )
    pdf.body_text(
        "This finding is consistent with neurolinguistic evidence that Broca's area is the primary region "
        "for syntactic reanalysis and structure-building. The transformer's middle layers, which are "
        "thought to handle compositional/syntactic representations, align most closely with human "
        "regression behavior."
    )

    # 3.4 Regression-Entropy Lag
    pdf.add_page()
    pdf.section_title("3.4 Regression-Entropy Lag", level=2)
    pdf.body_text(
        "A central question from the proposal (Section 4.1) is the temporal alignment: when humans "
        "physically look back at a difficult word, does the LLM show high entropy at that same word, "
        "or does the signal appear later in the processing window?"
    )

    lag_path = os.path.join(img_dir, "regression_entropy_lag_GPT2_DUNDEE.png")
    pdf.add_image_centered(lag_path, w=175,
                           caption="Figure 4: Regression-Entropy Lag - LLM entropy and human regression around peak-surprisal tokens")

    pdf.body_text(
        "Left panel: GPT-2 attention entropy is stable before the peak-surprisal word (~0.95-1.0) but spikes "
        "sharply at offsets +1 (1.19) and +2 (1.23). Right panel: Human regression-in probability peaks at "
        "the surprisal word itself (0.033) and decays rapidly. This reveals a fundamental temporal asymmetry:"
    )
    pdf.bullet("Humans react immediately: regression probability peaks at the difficult word itself")
    pdf.bullet("GPT-2 reacts with a 1-2 token lag: entropy peaks after the difficult word")
    pdf.body_text(
        "This lag is explained by GPT-2's autoregressive architecture: it can only 'realize' a word was "
        "difficult after processing it and seeing the next token(s). Humans, by contrast, can immediately "
        "initiate a backward saccade upon encountering difficulty. This finding has implications for "
        "understanding the differences between incremental human parsing and autoregressive LM processing."
    )

    # 3.5 Sentence Structure
    pdf.section_title("3.5 Sentence Structure Comparison", level=2)

    struct_path = os.path.join(img_dir, "structure_comparison_GPT2_DUNDEE.png")
    pdf.add_image_centered(struct_path, w=165,
                           caption="Figure 5: Correlation by sentence structure type")

    pdf.body_text(
        "Relative clause constructions show the strongest human-model alignment (surprisal-regression "
        "r = 0.21, entropy-regression r = 0.17), followed by active (r = 0.17, 0.13) and passive "
        "(r = 0.16, 0.10) sentences. This supports the proposal hypothesis that syntactically complex "
        "structures - which place greater demands on human parsing - produce stronger correspondence "
        "between LLM uncertainty and human back-tracking."
    )

    # 3.6 Regression Models
    pdf.add_page()
    pdf.section_title("3.6 Logistic Regression", level=2)
    pdf.body_text(
        "We model the probability of regression occurrence as a function of LLM-derived predictors "
        "and sentence structure using logistic regression (GLM with logit link)."
    )

    logistic_path = os.path.join(bayes_dir, "logistic_forest_plot.png")
    pdf.add_image_centered(logistic_path, w=165,
                           caption="Figure 6: Logistic regression coefficients with 95% confidence intervals")

    pdf.add_table(
        ["Predictor", "Beta (std)", "95% CI", "p-value", "Sig."],
        [
            ["surprisal", "+0.615", "[0.524, 0.706]", "< 0.0001", "***"],
            ["entropy_mean", "+0.682", "[0.550, 0.815]", "< 0.0001", "***"],
            ["struct_passive", "-0.059", "[-0.165, 0.048]", "0.279", "ns"],
            ["struct_rel_clause", "+0.008", "[-0.087, 0.103]", "0.875", "ns"],
        ],
        col_widths=[40, 30, 45, 40, 25]
    )
    pdf.body_text(
        "Both surprisal (beta = +0.615) and attention entropy (beta = +0.682) are independently significant "
        "predictors of regression occurrence at p < 0.0001. Notably, entropy has a slightly larger effect "
        "size than surprisal, suggesting that the 'breadth of attention' is an equally powerful predictor "
        "of human difficulty as prediction error. Sentence structure type is not significant after "
        "controlling for these information-theoretic measures."
    )

    pdf.section_title("3.7 Linear Regression (Reading Time)", level=2)

    linear_path = os.path.join(bayes_dir, "linear_forest_plot.png")
    pdf.add_image_centered(linear_path, w=165,
                           caption="Figure 7: Linear regression coefficients for reading time prediction")

    pdf.body_text(
        "The linear regression model for reading time yields R^2 = 0.019 with no individually significant "
        "predictors. This weak result is expected: the Dundee corpus reports averaged fixation durations "
        "across subjects, which compresses per-word variance. The logistic regression (regression "
        "occurrence) is the more appropriate dependent variable and shows strong results."
    )

    # 3.8 Reading Time scatter
    pdf.section_title("3.8 Reading Time Effects", level=2)
    rt_path = os.path.join(img_dir, "reading_time_effects_GPT2_DUNDEE.png")
    pdf.add_image_centered(rt_path, w=170,
                           caption="Figure 8: Reading time vs attention entropy (left) and surprisal (right)")

    # ================================================================
    # 4. DISCUSSION
    # ================================================================
    pdf.add_page()
    pdf.section_title("4. Discussion")

    pdf.section_title("4.1 Validation of Core Hypotheses", level=2)
    pdf.body_text(
        "Our results provide strong evidence for the Cognitive Symmetry hypothesis. Both GPT-2's surprisal "
        "and attention entropy significantly predict human regressive saccades, confirming that LLM "
        "uncertainty reflects genuine processing difficulty shared with human readers. The correlation "
        "magnitudes (r = 0.13-0.22) are consistent with prior work in computational psycholinguistics "
        "(Smith & Levy, 2013; Futrell et al., 2019)."
    )

    pdf.section_title("4.2 Novel Contributions", level=2)
    pdf.bold_text("Saccadic Heads:")
    pdf.body_text(
        "The identification of Head L1-H11 (r = 0.434) as a dominant saccadic head is a concrete "
        "interpretability finding. This head's position in Layer 1 suggests that GPT-2 can detect "
        "processing difficulty extremely early in its forward pass."
    )
    pdf.bold_text("Layer-to-Brain Mapping:")
    pdf.body_text(
        "The finding that middle layers (hypothesized Broca's area analogue) show the strongest regression "
        "prediction supports a functional correspondence between transformer layer depth and cortical "
        "processing stages. While this mapping is interpretive rather than causal, it provides a "
        "cognitively grounded framework for transformer interpretability."
    )
    pdf.bold_text("Regression-Entropy Lag:")
    pdf.body_text(
        "The 1-2 token lag in entropy response reveals a fundamental architectural difference: "
        "autoregressive models cannot 'look back' and must instead propagate difficulty signals forward "
        "through subsequent tokens, whereas humans can immediately initiate backward saccades."
    )

    # ================================================================
    # 5. NEXT STEPS (Final Submission)
    # ================================================================
    pdf.section_title("5. Next Steps (Final Submission)")
    pdf.bullet("Add LLaMA 3 as the high-tier model to complete the 2x2 design (GPT-2/LLaMA 3 x Dundee/GECO)")
    pdf.bullet("Incorporate the GECO corpus for cross-corpus validation")
    pdf.bullet("Expand to all 20 Dundee text files for greater statistical power")
    pdf.bullet("Implement full Bayesian regression with PyMC for hierarchical modeling across subjects")
    pdf.bullet("Add active/passive sentence classification using dependency parsing (spaCy)")
    pdf.bullet("Compare saccadic head patterns across model tiers to test the scaling hypothesis")

    # ================================================================
    # 6. REFERENCES
    # ================================================================
    pdf.section_title("6. References")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    refs = [
        "[1] Oh, B. D., and Schuler, W. (2025). Increasing alignment of large language models with language processing in the human brain. Nature Communications / arXiv.",
        "[2] Cop, U., Dirix, N., Drieghe, D., and Duyck, W. (2017). Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading. Behavior Research Methods, 49(2):602-615.",
        "[3] Kennedy, A. and Pynte, J. (2003). The Dundee Corpus. School of Psychology, The University of Dundee.",
        "[4] Levy, R. (2008). Expectation-based syntactic comprehension. Cognition, 106(3):1126-1177.",
        "[5] Smith, N. J. and Levy, R. (2013). The effect of word predictability on reading time is logarithmic. Cognition, 128(3):302-319.",
        "[6] Futrell, R., et al. (2019). Neural language models as psycholinguistic subjects: Representations of syntactic state. NAACL-HLT.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 5, ref)
        pdf.ln(1.5)

    # ================================================================
    # SAVE
    # ================================================================
    output_path = "results/Cognitive_Symmetry_Mid_Report.pdf"
    pdf.output(output_path)
    print(f"\nReport saved to: {output_path}")
    print(f"Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_report()
