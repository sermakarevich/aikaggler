# Fast ensemble of multi solutions + scores analysis

- **Author:** Kirderf
- **Votes:** 302
- **Ref:** kirderf/fast-ensemble-of-multi-solutions-scores-analysis
- **URL:** https://www.kaggle.com/code/kirderf/fast-ensemble-of-multi-solutions-scores-analysis
- **Last run:** 2025-01-20 13:17:00.493000

---

```python
!pip install /kaggle/input/hf-libraries/bitsandbytes/bitsandbytes-0.44.1-py3-none-manylinux_2_24_x86_64.whl
!pip install transformers==4.45.1
```

```python
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import glob
import os
from tabulate import tabulate

class PerplexityScorer:
    def __init__(self, model_path: str = '/kaggle/input/gemma-2/transformers/gemma-2-9b/2',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto'
        )
        self.model.eval()
        self.device = device
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity score exactly as in the official metric."""
        with torch.no_grad():
            # Add sequence boundary tokens
            text_with_special = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
            
            # Tokenize
            model_inputs = self.tokenizer(
                text_with_special,
                return_tensors='pt',
                add_special_tokens=False,
            )
            
            if 'token_type_ids' in model_inputs:
                model_inputs.pop('token_type_ids')
            
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            
            # Get model output
            output = self.model(**model_inputs, use_cache=False)
            logits = output['logits']
            
            # Shift logits and labels for calculating loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
            
            # Calculate token-wise loss
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Calculate perplexity
            sequence_loss = loss.sum() / len(loss)
            perplexity = torch.exp(sequence_loss).item()
            
            return perplexity

def load_submissions(base_dir: str = '.') -> List[Tuple[str, pd.DataFrame]]:
    """Load all submission files recursively from the directory."""
    submission_files = []
    
    # Walk through all directories recursively
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and 'submission' in file.lower():
                full_path = os.path.join(root, file)
                submission_files.append(full_path)
    
    if not submission_files:
        raise ValueError("No submission files found!")
    
    submissions = []
    print(f"\nLoading {len(submission_files)} submission files:")
    print("-" * 50)
    
    for file_path in submission_files:
        try:
            df = pd.read_csv(file_path)
            if 'id' in df.columns and 'text' in df.columns:
                print(f"Loading: {file_path}")
                submissions.append((file_path, df))
            else:
                print(f"Skipping {file_path} - Not a valid submission file (missing required columns)")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not submissions:
        raise ValueError("No valid submission files found!")
    
    return submissions

def evaluate_submissions(submissions: List[Tuple[str, pd.DataFrame]], 
                       scorer: PerplexityScorer) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate all submissions and track scores per text."""
    
    # Initialize storage for scores
    num_rows = len(submissions[0][1])  # Use the first submission's dataframe
    num_submissions = len(submissions)
    scores_matrix = np.zeros((num_rows, num_submissions))
    texts_matrix = [['' for _ in range(num_submissions)] for _ in range(num_rows)]
    
    print("\nEvaluating submissions:")
    print("-" * 50)
    
    submission_names = []
    
    # Calculate scores for each text in each submission
    for sub_idx, (file_path, submission) in enumerate(submissions):
        sub_name = file_path #os.path.basename(file_path)
        submission_names.append(sub_name)
        sub_scores = []
        
        print(f"\nSubmission {sub_idx + 1}/{num_submissions}: {sub_name}")
        
        for row_idx, text in enumerate(submission['text']):
            score = scorer.calculate_perplexity(text)
            scores_matrix[row_idx, sub_idx] = score
            texts_matrix[row_idx][sub_idx] = text
            sub_scores.append(score)
            print(f"Row {row_idx + 1}: Score = {score:.2f}")
        
        avg_score = np.mean(sub_scores)
        print(f"Average submission score: {avg_score:.2f}")
    
    # Create detailed scores DataFrame
    scores_df = pd.DataFrame(scores_matrix, columns=submission_names)
    
    # Find best texts and their sources
    best_texts = []
    best_scores = []
    best_sources = []
    
    for row_idx in range(num_rows):
        best_sub_idx = np.argmin(scores_matrix[row_idx])
        best_texts.append(texts_matrix[row_idx][best_sub_idx])
        best_scores.append(scores_matrix[row_idx, best_sub_idx])
        best_sources.append(submission_names[best_sub_idx])
    
    # Create summary dictionary
    summary = {
        'best_texts': best_texts,
        'best_scores': best_scores,
        'best_sources': best_sources,
        'submission_names': submission_names,
        'scores_matrix': scores_matrix
    }
    
    return scores_df, summary

def create_ensemble_submission(original_df: pd.DataFrame, 
                             summary: Dict, 
                             output_file: str):
    """Create final submission using best texts."""
    result_df = original_df.copy()
    result_df['text'] = summary['best_texts']
    
    # Print detailed results
    print("\nFinal Ensemble Results:")
    print("-" * 50)
    
    # Print submission averages first
    print("\nSubmission Averages:")
    for sub_idx, sub_name in enumerate(summary['submission_names']):
        avg_score = np.mean(summary['scores_matrix'][:, sub_idx])
        print(f"{sub_name}: {avg_score:.2f}")
    
    print("\nBest Texts Selected:")
    table_data = []
    for idx, (text, score, source) in enumerate(zip(
        summary['best_texts'], 
        summary['best_scores'], 
        summary['best_sources']
    )):
        table_data.append([
            idx,
            f"{text[:50]}..." if len(text) > 50 else text,
            f"{score:.2f}",
            source
        ])
    
    print(tabulate(table_data, 
                  headers=['Row', 'Best Text', 'Score', 'Source'],
                  tablefmt='grid'))
    
    final_avg_score = np.mean(summary['best_scores'])
    print(f"\nFinal Ensemble Average Score: {final_avg_score:.2f}")
    
    # Count usage of each submission
    source_counts = pd.Series(summary['best_sources']).value_counts()
    print("\nSubmission Usage Counts:")
    for source, count in source_counts.items():
        print(f"{source}: {count} texts")
    
    # Save final submission
    result_df.to_csv(output_file, index=False)
    print(f"\nEnsemble submission saved to {output_file}")
    
    # Also save detailed analysis
    analysis_file = 'submission_ensemble_analysis.csv'
    scores_df = pd.DataFrame({
        'Row': range(len(summary['best_texts'])),
        'Selected_Text': summary['best_texts'],
        'Score': summary['best_scores'],
        'Source': summary['best_sources']
    })
    scores_df.to_csv(analysis_file, index=False)
    print(f"Detailed analysis saved to {analysis_file}")

def main():
    # Initialize scorer
    scorer = PerplexityScorer()
    
    # Load all submission files from current directory and subdirectories
    submissions = load_submissions('/kaggle/input')
    
    # Evaluate all submissions
    scores_df, summary = evaluate_submissions(submissions, scorer)
    
    # Save detailed scores
    scores_df.to_csv('submission_scores_analysis.csv')
    print("\nDetailed scores saved to submission_scores_analysis.csv")
    
    # Create and save ensemble submission
    create_ensemble_submission(submissions[0][1], summary, 'submission.csv')

if __name__ == "__main__":
    main()
```