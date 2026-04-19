import os
from data_utils import load_geco_dataset, load_dundee_dataset, preprocess_for_model
from llm_pipeline import GPT2Evaluator, BertEvaluator
from analysis import align_and_correlate, plot_saccadic_heads, plot_regression_entropy_lag, save_text_results, plot_reading_time_effects

def process_dataset(dataset_name, dataset_df, evaluators):
    print(f"=== Processing [{dataset_name}] ===")
    human_data = preprocess_for_model(dataset_df)
    
    # Evaluate across sentences
    for evaluator_name, evaluator in evaluators.items():
        print(f"\n  -> Running {evaluator_name} on {dataset_name}...")
        llm_data_metrics = []
        for i, item in enumerate(human_data):
            sentence = item["sentence"]
            # To speed up demonstration/testing, limit to 20 sentences if the dataset is large
            if i >= 20:
                break
            metrics = evaluator.evaluate_sentence(sentence)
            llm_data_metrics.append({
                "sentence": sentence,
                "metrics": metrics
            })
            
        print(f"  -> Aligning and correlating {evaluator_name} vs {dataset_name}...")
        # Make sure human_data size matches llm_data_metrics for alignment
        aligned_human = human_data[:len(llm_data_metrics)]
        df, corrs = align_and_correlate(aligned_human, llm_data_metrics)
        
        save_text_results(corrs, f"{evaluator_name}", f"{dataset_name}")
        plot_saccadic_heads(df, f"{evaluator_name}_{dataset_name}")
        plot_regression_entropy_lag(df, f"{evaluator_name}_{dataset_name}")
        plot_reading_time_effects(df, f"{evaluator_name}_{dataset_name}")
        print("-" * 40)

def main():
    print("Initializing Cognitive Symmetry Pipeline...")
    
    # Create required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Load Models
    evaluators = {
        "GPT2": GPT2Evaluator("gpt2"),
        "BERT": BertEvaluator("bert-base-uncased")
    }
    
    
    # Process Dundee dataset
    dundee_df = load_dundee_dataset("dundee")
    process_dataset("DUNDEE", dundee_df, evaluators)
    
    print("Pipeline Complete! Check 'results' folder for output artifacts.")

if __name__ == "__main__":
    main()
