from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
# Custom prompt for summarization
from retrival import basic_retrieval
import pandas as pd
from cot import do_cot

openai_key = "<YOUR-API>"
df = pd.read_csv("/workspace/data/qa_netflix_pairs.csv")
questions = list(df.Question)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=openai_key)
generated_output=[]

for i in range(len(questions)):
    summary_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:"""
    )
    
    # Retrieve data from the graph
    query = questions[i]
    #find sub-questions
    sub_queries_l, _ = do_cot(query)
    
    retrieved_data_l = []
    if sub_queries_l is not None:
        queries = [query]+sub_queries_l
    else:
        queries = [query]
    for j in range(len(queries)):
        try:
            retrieved_data=basic_retrieval(queries[j])
            retrieved_data_l.append(retrieved_data)
        except:
            retrieved_data=basic_retrieval(queries[0])
            retrieved_data_l.append(retrieved_data)
            break
    
    # Use the LLM to generate a summary
    summary = llm.invoke(summary_prompt.format(context=retrieved_data_l, question=query))
    generated_output.append(summary.content)
    print(f"#########Q_{i}#########")
    print("My Question is:")
    print(query)
    print("My Sub Questions are:")
    print(sub_queries_l)
    print("My Retrieved Data is:")
    print(retrieved_data_l)
    print("My Generated Answer is:")
    print(summary.content)
    print("\n")
generated_dict = {"Question":questions, "gen_output":generated_output}
df_generated = pd.DataFrame(generated_dict)
df_generated.to_csv("/workspace/data/qa_netflix_pair_generated_new_test_v3.csv", index=False)