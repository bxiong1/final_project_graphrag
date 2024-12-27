# final_project_graphrag

# GraphRAG Final Project

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

---

## Installation

Step-by-step instructions to install and set up the project locally.

1. Clone the repository:
   ```bash
   git clone https://github.com/bxiong1/final_project_graphrag.git
   ```
2. Install necessary packages
   ```pip install -r requirements.txt
   ```
3. Replace the API Key in the following files:
   ```
   cot.py --line 7
   generation_new.py --line 8
   generation_plain.py --line 8
   gpt_judge.py --line 6
   retrieval.py --line 11
   ```
4. Replace the working directory with your own to all the files under the project
---


## Usage
There are two types of usages:
1. Express Usage (Recommend)
2. Full Usage

# Express Usage
1. We have already complete building the graph on Neo4j and obtained all the relevant retrieved and generated answer to the project (under [file](/data/qa_netflix_pair_generated_new_test.csv)), thus the first command you have to run is:
   ```
   python evaluate.py #This will produce Similarity, Groundness, Relevancy and Coherence
   python more_evaluation.py #This will produce Bert_Scorer Precision, Recall and F1
   python gpt_judge.py #This produce the results for GPT evaluation
   ```
2. The average results on these metric will be printed on the terminal shell.

# Full Usage
1. Run building graph (This could take a long time)
   ```
   python build_graph_netflix.py
   ```
2. Run retrieval and generation
   ```
   python generation_new.py
   ```
3. To obtain the baseline generation
   ```
   python generation_plain.py
   ```
4. The results will be recorded under the [/data](/data) folder, Then run the evaluation again (indicated in Express Usage) to obtain the evaluation performance.

## Contribution
Our Team members that contribute to this project are: Shanyi Li, Zhiyuan He, Zehua Pei, Chen Xiong
