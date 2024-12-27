import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score
import pandas as pd
import csv

# Make sure to download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')

def compute_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Compute BLEU score for the hypothesis against references.
    """
    tokenized_references = [ref.split() for ref in references]
    tokenized_hypothesis = hypothesis.split()
    bleu_score = sentence_bleu(
        tokenized_references,
        tokenized_hypothesis,
        weights=weights,
        smoothing_function=SmoothingFunction().method1
    )
    return bleu_score


def compute_rouge(references, hypothesis):
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hypothesis) for ref in references]

    # Average the scores across all references
    rouge1 = sum(score['rouge1'].fmeasure for score in scores) / len(scores)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores) / len(scores)
    rougeL = sum(score['rougeL'].fmeasure for score in scores) / len(scores)

    return {
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL
    }


def compute_meteor(references, hypothesis):
    """
    Compute METEOR score.
    """
    # NLTK's METEOR expects a list of references
    meteor = meteor_score(references, hypothesis)
    return meteor


def compute_bert_score(references, hypothesis, model_type="bert-base-uncased"):
    """
    Compute BERTScore using pre-trained BERT model.
    :param references: List of reference strings
    :param hypothesis: Single hypothesis string
    :param model_type: Pre-trained model used for BERTScore
    """
    # Wrap references in an outer list to treat them as one multi-reference set
    P, R, F1 = score([hypothesis], [references], model_type=model_type, verbose=False)
    return {
        "BERTScore_Precision": P.item(),
        "BERTScore_Recall": R.item(),
        "BERTScore_F1": F1.item()
    }

if __name__ == "__main__":
    # Load data from CSV files
    df_ori = pd.read_csv("/workspace/data/qa_netflix_pairs.csv")
    df_generated = pd.read_csv("//workspace/data/qa_netflix_pair_generated_new_test.csv")

    # Extract questions, ground truth answers, and generated answers
    questions = list(df_ori.Question)
    gt_answer = list(df_ori.Answer)
    gen_answer = list(df_generated['gen_output'])

    results = []  # To store metric results for each example

    # Loop through all question-answer pairs
    for i in range(len(questions)):
        question = questions[i]
        ground_truth = gt_answer[i]
        generated_answer = gen_answer[i]

        # Compute metrics for the current pair
        bleu = compute_bleu([ground_truth], generated_answer)
        rouge = compute_rouge([ground_truth], generated_answer)
        #meteor = compute_meteor([ground_truth], generated_answer)
        bert_score = compute_bert_score([ground_truth], generated_answer)

        # Combine all scores into a single dictionary
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "BLEU": bleu,
            "ROUGE-1": rouge["ROUGE-1"],
            "ROUGE-2": rouge["ROUGE-2"],
            "ROUGE-L": rouge["ROUGE-L"],
            "BERTScore_Precision": bert_score["BERTScore_Precision"],
            "BERTScore_Recall": bert_score["BERTScore_Recall"],
            "BERTScore_F1": bert_score["BERTScore_F1"],
        }

        # Append the result to the list
        results.append(result)

    # Calculate average scores for each metric
    average_scores = {}
    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1"]
    for metric in metrics:
        average_scores[metric] = sum(result[metric] for result in results) / len(results)

    # Write the evaluation results to a new CSV file
    output_csv = "/workspace/data/qa_netflix_pair_generated_plain_more_eval.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as out_file:
        fieldnames = ["question", "ground_truth", "generated_answer"] + metrics
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluation completed. Results saved to {output_csv}")
    print("Average Scores:")
    for metric, avg_score in average_scores.items():
        print(f"{metric}: {avg_score:.4f}")