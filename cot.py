import openai
import json

def do_cot(question):
    # Set up OpenAI API key (replace YOUR_API_KEY with your actual key)

    client = openai.OpenAI(api_key="<YOUR-API>")
    # Define the updated system prompt
    system_prompt = """
    You are a Netflix Movie Knowledge Assistant. Your role is to analyze complex queries related to Netflix movies and TV shows and break them down into smaller, logical subqueries that can be directly converted into Cypher queries. You should:
    
    1. Understand the user's query and identify its key components.
    2. Break the query into clear, step-by-step subqueries that include specific details required to construct Cypher queries (e.g., nodes, relationships, and properties).
    3. Ensure each subquery is concise, specific, and follows a structure that can be easily translated into Cypher queries.
    4. Provide the final subqueries **only in JSON format**, where each subquery is a numbered key, and the value is the subquery in natural language.
    
    Do not generate actual Cypher queries — only provide the subqueries in JSON format. Ensure that all subqueries are logically organized, structured, and accurate. If the query is ambiguous, ask clarifying questions instead of making incorrect assumptions.
    
    Here is an example of how you should respond:
    
    ---
    
    **Input Query:**  
    "Find all Netflix movies directed by Christopher Nolan that are in the Action genre and have a rating above 8.5."
    
    **Final Subqueries (in JSON format):**  
    {
        "1": "Find all movies where the director's name is Christopher Nolan.",
        "2": "Filter these movies to include only those that belong to the Action genre.",
        "3": "Further filter these Action movies to include only those with a rating above 8.5."
    }
    
    ---
    
    Always ensure the subqueries contain sufficient detail to construct Cypher queries and provide them in this JSON structure. Exclude any additional reasoning or explanations.
    """
    
    # Example user query
    user_query = question
    
    # Make the API call with the updated prompt
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature = 0.0,
        top_p = 1.0,
    )
    
    # Extract and print the JSON response
    response_text = response.choices[0].message.content
    for ind in range(10):
        try:
            # Parse JSON if the response is in JSON format
            subqueries_json = json.loads(response_text.strip())
            key_l = list(subqueries_json.keys())
            sub_questions = []
            for i in range(len(key_l)):
                sub_questions.append(subqueries_json[key_l[i]])
            return sub_questions, subqueries_json
        except json.JSONDecodeError:
            print("Error:")
            print(response_text)
            continue
    return None, None
    