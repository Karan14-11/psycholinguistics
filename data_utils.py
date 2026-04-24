import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_dundee_dataset(dundee_dir="dundee", num_texts=5, max_sentences_per_text=100):
    """
    Parses the raw Dundee Corpus across multiple text files (tx01–tx20)
    and all 10 subjects (sa–sj) per text.

    Returns a DataFrame with columns:
        sentence, word_idx, word, regression_in_prob, regression_out_prob,
        reading_time, text_id, sentence_id_global
    """
    if not os.path.isdir(dundee_dir):
        print(f"[Dundee] Directory '{dundee_dir}' not found.")
        return pd.DataFrame()

    all_data = []
    global_sentence_offset = 0

    for text_num in range(1, num_texts + 1):
        tx_file = os.path.join(dundee_dir, f"tx{text_num:02d}wrdp.dat")
        if not os.path.exists(tx_file):
            print(f"[Dundee] Text file {tx_file} not found, skipping.")
            continue

        print(f"Parsing Dundee Text {text_num:02d}...")

        # ---- 1. Parse the word-position file to get word list ----
        words_data = []
        current_sentence = 1
        with open(tx_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 12:
                    word = parts[0].strip()
                    wnum = int(parts[11])  # global token id within this text
                    words_data.append((wnum, word, current_sentence))
                    if word.endswith('.') or word.endswith('?') or word.endswith('!'):
                        current_sentence += 1

        df_words = pd.DataFrame(words_data, columns=["WNUM", "word", "sentence_id"])

        # ---- 2. Parse eye-tracking data across all subjects for this text ----
        subject_files = sorted(glob.glob(
            os.path.join(dundee_dir, f"s*{text_num:02d}ma1p.dat")
        ))
        num_subjects = len(subject_files)
        if num_subjects == 0:
            print(f"  No subject files found for text {text_num:02d}, skipping.")
            continue

        # Initialise counters
        reg_in_counts = {wnum: 0 for wnum in df_words["WNUM"]}
        reg_out_counts = {wnum: 0 for wnum in df_words["WNUM"]}
        rt_sums = {wnum: 0.0 for wnum in df_words["WNUM"]}
        rt_counts = {wnum: 0 for wnum in df_words["WNUM"]}
        fixation_counts = {wnum: 0 for wnum in df_words["WNUM"]}

        for sf in subject_files:
            prev_wnum = -1
            with open(sf, 'r', encoding='latin-1') as f:
                next(f)  # skip header
                for line in f:
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            wnum = int(parts[6])   # WNUM column
                            fdur = int(parts[7])   # FDUR (fixation duration)

                            if wnum > 0:
                                if wnum in rt_sums:
                                    rt_sums[wnum] += fdur
                                    rt_counts[wnum] += 1
                                    fixation_counts[wnum] += 1

                                if prev_wnum > 0:
                                    if wnum < prev_wnum:
                                        # Regression IN to wnum (going backward)
                                        if wnum in reg_in_counts:
                                            reg_in_counts[wnum] += 1
                                        # Regression OUT from prev_wnum
                                        if prev_wnum in reg_out_counts:
                                            reg_out_counts[prev_wnum] += 1

                            prev_wnum = wnum
                        except ValueError:
                            pass

        # ---- 3. Build per-word DataFrame for this text ----
        sentences_processed = 0
        for sent_id, group in df_words.groupby("sentence_id"):
            if sentences_processed >= max_sentences_per_text:
                break

            words_in_sent = group["word"].tolist()
            # Skip very short sentences (< 3 words) or very long ones (> 60 words)
            if len(words_in_sent) < 3 or len(words_in_sent) > 60:
                continue

            sent_str = " ".join(words_in_sent)

            for i, row in enumerate(group.itertuples()):
                wnum = row.WNUM
                p_in = reg_in_counts.get(wnum, 0) / num_subjects if num_subjects > 0 else 0
                p_out = reg_out_counts.get(wnum, 0) / num_subjects if num_subjects > 0 else 0
                avg_rt = rt_sums.get(wnum, 0) / rt_counts[wnum] if rt_counts.get(wnum, 0) > 0 else 0

                all_data.append({
                    "sentence": sent_str,
                    "word_idx": i,
                    "word": row.word,
                    "regression_in_prob": p_in,
                    "regression_out_prob": p_out,
                    "reading_time": avg_rt,
                    "text_id": text_num,
                    "sentence_id_global": global_sentence_offset + sent_id
                })

            sentences_processed += 1

        global_sentence_offset += current_sentence
        print(f"  Text {text_num:02d}: {sentences_processed} sentences parsed, "
              f"{num_subjects} subjects.")

    df = pd.DataFrame(all_data)
    print(f"\n[Dundee] Total: {len(df)} word-tokens across "
          f"{df['sentence_id_global'].nunique() if len(df) > 0 else 0} sentences.")
    return df


def preprocess_for_model(df):
    """
    Groups word-level rows back into full sentences for LM consumption.
    Returns a list of dicts: {"sentence": str, "metrics": list_of_dicts}
    """
    sentences = []
    grouped = df.groupby("sentence", sort=False)
    for sent, group in grouped:
        metrics = []
        for _, row in group.iterrows():
            metrics.append({
                "word": row["word"],
                "word_idx": row["word_idx"],
                "regression_in_prob": row["regression_in_prob"],
                "regression_out_prob": row["regression_out_prob"],
                "reading_time": row.get("reading_time", 0),
                "text_id": row.get("text_id", 0),
                "sentence_id_global": row.get("sentence_id_global", 0),
            })
        sentences.append({
            "sentence": sent,
            "metrics": metrics
        })
    return sentences
