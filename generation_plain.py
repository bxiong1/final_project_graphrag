from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
# Custom prompt for summarization
from retrival import basic_retrieval
import pandas as pd


openai_key = "<YOUR-API>"
df = pd.read_csv("/workspace/data/qa_netflix_pairs.csv")
questions = list(df.Question)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=openai_key)
generated_output=[]

for i in range(len(questions)):
    summary_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
    Question: {question}
    
    Helpful Answer:"""
    )
    
    # Retrieve data from the graph
    query = questions[i]
    
    # Use the LLM to generate a summary
    summary = llm.invoke(summary_prompt.format(question=query))
    generated_output.append(summary.content)
    print(f"#########Q_{i}#########")
    print("My Question is:")
    print(query)
    print("My Generated Answer is:")
    print(summary.content)
    print("\n")
generated_dict = {"Question":questions, "gen_output":generated_output}
df_generated = pd.DataFrame(generated_dict)
df_generated.to_csv("/workspace/data/qa_netflix_pair_generated_plain.csv", index=False)