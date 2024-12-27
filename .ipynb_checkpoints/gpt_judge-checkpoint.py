import openai
import csv
import pandas as pd

# Set your OpenAI API key
client = openai.OpenAI(api_key="<YOUR-API>")

def gpt_judge(question, ground_truth, generated_answer, model="gpt-4o"):
    """
    Use GPT to evaluate the similarity between the ground truth and the generated answer.
    :param question: The question being answered.
    :param ground_truth: The ground truth answer.
    :param generated_answer: The generated answer.
    :param model: OpenAI GPT model to use.
    :return: A similarity score and GPT's explanation.
    """
    # Define the prompt for GPT
    prompt = f"""
You are an evaluation assistant. Your task is to judge how close a generated answer is to the ground truth answer for a given question. 
Provide a similarity score between 0 and 1, where:
- 1 means the generated answer is identical or fully aligned with the ground truth.
- 0 means the generated answer is completely incorrect or unrelated.
Additionally, explain why you assigned the score.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Please provide your evaluation in the following format:
Score: [similarity_score]
Explanation: [your explanation]
    """

    # Make the API call to GPT
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates text similarity."},
                {"role": "user", "content": prompt},
            ],
            temperature=0  # Set to 0 for consistent scoring
        )
        # Extract GPT's response
        content = response.choices[0].message.content
        
        # Parse the score from GPT's response
        score_line = [line for line in content.split("\n") if line.startswith("Score:")][0]
        score = float(score_line.split(":")[1].strip())
        
        # Extract the explanation
        explanation_line = [line for line in content.split("\n") if line.startswith("Explanation:")][0]
        explanation = explanation_line.split(":")[1].strip()
        
        return score, explanation

    except Exception as e:
        print(f"Error during GPT evaluation: {e}")
        return None, "Error occurred during GPT evaluation."

if __name__ == "__main__":
    # Load data from CSV files
    df_ori = pd.read_csv("/workspace/data/qa_netflix_pairs.csv")
    df_generated = pd.read_csv("/workspace/data/qa_netflix_pair_generated_new_test.csv")

    # Extract questions, ground truth answers, and generated answers
    questions = list(df_ori.Question)
    gt_answer = list(df_ori.Answer)
    gen_answer = list(df_generated['gen_output'])

    results = []  # To store metric results for each example
    similarity_scores = []  # To store similarity scores for averaging

    # Loop through all question-answer pairs
    for i in range(len(questions)):
        question = questions[i]
        ground_truth = gt_answer[i]
        generated_answer = gen_answer[i]

        # Use GPT to evaluate the similarity
        score, explanation = gpt_judge(question, ground_truth, generated_answer)

        # Append the result to the list
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "GPT_Similarity_Score": score,
            "GPT_Explanation": explanation,
        }

        # Add score to similarity_scores for averaging
        if score is not None:
            similarity_scores.append(score)

        results.append(result)

    # Calculate the average similarity score
    average_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

    # Write the evaluation results to a new CSV file
    output_csv = "/workspace/data/qa_netflix_pair_generated_new_test_gpt_eval.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as out_file:
        fieldnames = ["question", "ground_truth", "generated_answer", "GPT_Similarity_Score", "GPT_Explanation"]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluation completed. Results saved to {output_csv}")
    print(f"Average GPT Similarity Score: {average_score:.4f}")