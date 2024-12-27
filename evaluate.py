import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import csv

class Evaluation:
    def __init__(self, model_name="gpt-3.5"):
        self.model_name = model_name
        self.embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

    def cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        """
        if all(v == 0 for v in vec1) or all(v == 0 for v in vec2):
            return 0.0
        return 1 - cosine(vec1, vec2)

    def embed_text(self, text):
        """
        Generate embeddings for a given text using a transformer-based model.
        """
        embeddings = self.embedding_model(text)
        # Reduce the dimensionality of embeddings
        return np.mean(embeddings[0], axis=0)

    def evaluate_similarity(self, answer, source):
        """
        Evaluate similarity between ground truth and generated answer using cosine similarity.
        """
        vec1 = self.embed_text(answer)
        vec2 = self.embed_text(source)
        if not vec1.any() or not vec2.any():
            raise ValueError("One of the embedding vectors is empty.")
        return self.cosine_similarity(vec1, vec2)

    def evaluate_coherence(self, question, answer):
        """
        Evaluate coherence between the question and the generated answer.
        """
        # Use TF-IDF vectorizer to evaluate semantic closeness
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([question, answer])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    def groundedness(self, question, source, answer):
        """
        Evaluate groundedness by checking if the answer overlaps semantically with the source.
        """
        vec1 = self.embed_text(answer)
        vec2 = self.embed_text(source)
        return self.cosine_similarity(vec1, vec2)

    def context_relevancy(self, question, source, answer):
        """
        Evaluate context relevancy by comparing the source and the answer.
        """
        vec1 = self.embed_text(source)
        vec2 = self.embed_text(answer)
        return self.cosine_similarity(vec1, vec2)

    def evaluate_toxicity(self, text):
        """
        Evaluate toxicity of a given text using a pre-trained toxicity detection model.
        """
        toxicity_pipeline = pipeline("text-classification", model="unitary/toxic-bert")
        result = toxicity_pipeline(text)
        return result[0]["score"], result[0]["label"]

    def evaluate_bias(self, text):
        """
        Evaluate bias using a simple keyword-based approach.
        """
        bias_keywords = ['gender', 'race', 'ethnicity', 'religion']
        bias_count = sum(keyword in text.lower() for keyword in bias_keywords)
        if bias_count > 0:
            return 1.0, "Bias Detected"
        return 0.0, "No Bias Detected"

    def evaluate(self, question, source, answer):
        """
        Evaluate multiple metrics for the generated answer.
        """
        similarity_score = self.evaluate_similarity(answer, source)
        coherence_score = self.evaluate_coherence(question, answer)
        groundedness_score = self.groundedness(question, source, answer)
        relevancy_score = self.context_relevancy(question, source, answer)
        toxicity_score, toxicity_label = self.evaluate_toxicity(answer)
        bias_score, bias_label = self.evaluate_bias(answer)

        return {
            "similarity": similarity_score,
            "coherence": coherence_score,
            "groundedness": groundedness_score,
            "context_relevancy": relevancy_score,
            "toxicity": (toxicity_score, toxicity_label),
            "bias": (bias_score, bias_label)
        }

if __name__ == "__main__":
    df_ori = pd.read_csv("/workspace/data/qa_netflix_pairs.csv")
    df_generated = pd.read_csv("/workspace/data/qa_netflix_pair_generated_new_test.csv")
    # Example usage of the Evaluation class
    eval_tool = Evaluation()

    questions = list(df_ori.Question)
    gt_answer = list(df_ori.Answer)
    gen_answer = list(df_generated['gen_output'])
    # Example inputs
    results = []
    total_scores = {
            "similarity": 0.0,
            "coherence": 0.0,
            "groundedness": 0.0,
            "context_relevancy": 0.0,
            "toxicity": 0.0,
            "bias": 0.0
    }
    row_count = 0
    
    for i in range(len(questions)):
        question = questions[i]
        source = gt_answer[i]
        generated_answer = gen_answer[i]
        
        # Perform evaluation
        evaluation_result = eval_tool.evaluate(question, source, generated_answer)
        
        results.append({
                "question": question,
                "ground_truth": source,
                "generated_answer": generated_answer,
                **evaluation_result
            })

            # Update totals for each metric
        total_scores["similarity"] += evaluation_result["similarity"]
        total_scores["coherence"] += evaluation_result["coherence"]
        total_scores["groundedness"] += evaluation_result["groundedness"]
        total_scores["context_relevancy"] += evaluation_result["context_relevancy"]
        total_scores["toxicity"] += evaluation_result["toxicity"][0]  # Only add toxicity score
        total_scores["bias"] += evaluation_result["bias"][0]  # Only add bias score
        row_count += 1

        # Compute averages
        average_scores = {metric: total / row_count for metric, total in total_scores.items()}

    # Write the evaluation results to a new CSV file
    output_csv = "/workspace/data/evaluation_results_new_test.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as out_file:
        fieldnames = ["question", "ground_truth", "generated_answer"] + list(results[0].keys())[3:]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluation completed. Results saved to {output_csv}")
    print("Average Scores:")
    for metric, avg_score in average_scores.items():
        print(f"{metric}: {avg_score:.4f}")