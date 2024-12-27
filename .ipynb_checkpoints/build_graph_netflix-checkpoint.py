# Import necessary libraries
import os
import csv
import warnings
from neo4j import GraphDatabase

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
CONFIG = {
    "NEO4J_URI": "neo4j+s://6bb08534.databases.neo4j.io",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "LzdFRD0j0fqPVk2tOsfCmN037fjD18P9_-550ULIi6c",
    "CSV_FILE_PATH": "/workspace/data/netflix_titles.csv"
}

# Initialize Neo4j driver
def get_neo4j_driver(config):
    return GraphDatabase.driver(config["NEO4J_URI"], auth=(config["NEO4J_USERNAME"], config["NEO4J_PASSWORD"]))

# Query to clear all data in the database
CLEAR_DATABASE_QUERY = """
MATCH (n)
DETACH DELETE n;
"""

# Load data from CSV into Neo4j
def load_movies_from_csv(driver, csv_path):
    def insert_movie(tx, row):
        tx.run("""
            CREATE (m:Movie {
                id: $show_id,
                type: $type,
                title: $title,
                director: $director,
                cast: $cast,
                country: $country,
                date_added: $date_added,
                release_year: $release_year,
                rating: $rating,
                duration: $duration,
                listed_in: $listed_in,
                description: $description
            })
        """, **row)

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        with driver.session() as session:
            for row in reader:
                # Pass the row explicitly to the transaction function
                session.write_transaction(insert_movie, row)

    print("Movies loaded into the database successfully.")

# Queries to build relationships and enrich the graph
RELATIONSHIP_QUERIES = {
    "ACTED_IN": """
        MATCH (m:Movie)
        WHERE m.cast IS NOT NULL
        UNWIND split(m.cast, ',') AS actor
        MERGE (p:Person {name: trim(actor)})
        MERGE (p)-[:ACTED_IN]->(m);
    """,

    "IN_CATEGORY": """
        MATCH (m:Movie)
        WHERE m.listed_in IS NOT NULL
        UNWIND split(m.listed_in, ',') AS category
        MERGE (c:Category {name: trim(category)})
        MERGE (m)-[:IN_CATEGORY]->(c);
    """,

    "TYPED_AS": """
        MATCH (m:Movie)
        WHERE m.type IS NOT NULL
        MERGE (t:Type {type: m.type})
        MERGE (m)-[:TYPED_AS]->(t);
    """,

    "DIRECTED": """
        MATCH (m:Movie)
        WHERE m.director IS NOT NULL
        MERGE (d:Person {name: m.director})
        MERGE (d)-[:DIRECTED]->(m);
    """,

    "LOCATED_IN": """
        MATCH (m:Movie)
        WHERE m.country IS NOT NULL
        MERGE (c:Country {name: trim(m.country)})
        MERGE (m)-[:WHERE]->(c);
    """,

    "WORK_WITH": """
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
        MERGE (p)-[:WORK_WITH]->(p2);
    """,

    "CREATED_ON": """
        MATCH (m:Movie)
        WHERE m.release_year IS NOT NULL
        WITH m, toInteger(m.release_year) AS year
        MERGE (y:Year {value: year})
        MERGE (m)-[:CREATED_ON]->(y);
    """,

    "TIME_TREE": """
        WITH range(2012, 2019) AS years
        FOREACH(year IN years |
          CREATE (y:Year {value: year})
        );

        MATCH (y1:Year), (y2:Year)
        WHERE y1.value + 1 = y2.value
        MERGE (y1)-[:NEXT]->(y2);
    """,

    "DELETE_UNUSED_PROPERTIES": """
        MATCH (m:Movie)
        SET m.country = null, m.category = null, m.type = null,
            m.director = null, m.cast = null;
    """
}

# Execute a list of queries in Neo4j
def execute_queries(driver, queries):
    with driver.session() as session:
        for name, query in queries.items():
            print(f"Executing query: {name}")
            session.run(query)

# Main function to orchestrate the workflow
def main():
    driver = get_neo4j_driver(CONFIG)

    try:
        # Step 1: Clear existing data
        print("Clearing database...")
        with driver.session() as session:
            session.run(CLEAR_DATABASE_QUERY)

        # # Step 2: Load data from CSV
        print("Loading movies from CSV...")
        load_movies_from_csv(driver, CONFIG["CSV_FILE_PATH"])

        # Step 3: Execute relationship-building queries
        print("Building relationships...")
        execute_queries(driver, RELATIONSHIP_QUERIES)

        print("Database setup complete.")

    finally:
        driver.close()

# Run the main function
if __name__ == "__main__":
    main()