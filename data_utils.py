import os
import pandas as pd
import numpy as np

def load_geco_dataset(filepath="data/geco.csv"):
    """
    Loads GECO dataset. If the file is not present, it generates a simulated garden-path dataset
    that mimics the regression rates typical of such sentences (~30%).
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df
    return _generate_dummy_data("geco")

def load_dundee_dataset(filepath="dundee"):
    """
    Parses the raw Dundee Corpus from the folder.
    """
    if os.path.isdir(filepath):
        tx_file = os.path.join(filepath, "tx01wrdp.dat")
        if not os.path.exists(tx_file):
            return _generate_dummy_data("dundee")
            
        print("Parsing Dundee corpus (Text 1)...")
        # 1. Parse text to get word list
        words_data = [] # list of (wnum, word, sentence_id)
        current_sentence = 1
        with open(tx_file, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 12:
                    word = parts[0]
                    wnum = int(parts[11]) # global_token_id
                    word = word.strip()
                    words_data.append((wnum, word, current_sentence))
                    # Check for sentence boundary (very rough approach)
                    if word.endswith('.') or word.endswith('?') or word.endswith('!'):
                        current_sentence += 1
                        
        df_words = pd.DataFrame(words_data, columns=["WNUM", "word", "sentence_id"])
        
        # 2. Parse eye-tracking data for Text 1 across subjects
        # We look for sa01, sb01, sc01... up to sj01
        import glob
        subject_files = glob.glob(os.path.join(filepath, "s*01ma1p.dat"))
        
        reg_in_counts = {wnum: 0 for wnum in df_words["WNUM"]}
        reg_out_counts = {wnum: 0 for wnum in df_words["WNUM"]}
        rt_sums = {wnum: 0 for wnum in df_words["WNUM"]}
        rt_counts = {wnum: 0 for wnum in df_words["WNUM"]}
        num_subjects = len(subject_files)
        
        for sf in subject_files:
            prev_wnum = -1
            with open(sf, 'r') as f:
                next(f) # skip header
                for line in f:
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            wnum = int(parts[6]) # WNUM column
                            fdur = int(parts[7]) # FDUR column (First Fixation Duration/Reading Time)
                            
                            if wnum > 0:
                                # Track reading time
                                if wnum in rt_sums:
                                    rt_sums[wnum] += fdur
                                    rt_counts[wnum] += 1
                                    
                                if prev_wnum > 0:
                                    if wnum < prev_wnum:
                                        # Regression IN to wnum
                                        if wnum in reg_in_counts:
                                            reg_in_counts[wnum] += 1
                                        # Regression OUT from prev_wnum
                                        if prev_wnum in reg_out_counts:
                                            reg_out_counts[prev_wnum] += 1
                            prev_wnum = wnum
                        except ValueError:
                            pass
                            
        # Compute probabilities
        data = []
        for sent_id, group in df_words.groupby("sentence_id"):
            sent_str = " ".join(group["word"].tolist())
            for i, row in enumerate(group.itertuples()):
                wnum = row.WNUM
                p_in = reg_in_counts[wnum] / num_subjects if num_subjects > 0 else 0
                p_out = reg_out_counts[wnum] / num_subjects if num_subjects > 0 else 0
                avg_rt = rt_sums[wnum] / rt_counts[wnum] if rt_counts[wnum] > 0 else 0
                
                data.append({
                    "sentence": sent_str,
                    "word_idx": i,
                    "word": row.word,
                    "regression_in_prob": p_in,
                    "regression_out_prob": p_out,
                    "reading_time": avg_rt
                })
            # Limit to 50 sentences for demonstration
            if sent_id >= 50:
                break
                
        return pd.DataFrame(data)
    return _generate_dummy_data("dundee")

def _generate_dummy_data(corpus_name):
    print(f"[{corpus_name}] Warning: Dataset file not found. Generating simulated garden-path sentences with regression metrics...")
    
    sentences = [
        "The old man the boat .",
        "The complex houses married and single soldiers and their families .",
        "The horse raced past the barn fell .",
        "While the man hunted the deer ran into the woods .",
        "The cotton clothing is made of grows in Mississippi ."
    ]
    
    # We will synthetically assign high regression-in probabilities to the disambiguating words.
    data = []
    
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            # Baseline probabilities
            reg_in = np.random.uniform(0.05, 0.15)
            reg_out = np.random.uniform(0.05, 0.15)
            
            # Artificial spikes for garden-path disambiguating words
            if word in ["man", "houses", "fell", "ran", "grows"]:
                reg_in = np.random.uniform(0.25, 0.45) # Spike ~30%+
                reg_out = np.random.uniform(0.25, 0.45)
                
            data.append({
                "sentence": sentence,
                "word_idx": i,
                "word": word,
                "regression_in_prob": reg_in,
                "regression_out_prob": reg_out,
            })
            
    return pd.DataFrame(data)

def preprocess_for_model(df):
    """
    Groups the word-level rows back into full sentences for LM consumption.
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
                "reading_time": row.get("reading_time", 0)
            })
        sentences.append({
            "sentence": sent,
            "metrics": metrics
        })
    return sentences
