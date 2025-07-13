import os
from dotenv import load_dotenv
from lib.vector_db import VectorStoreManager, GamesLoaderService
from lib.document import GameDocument

# Load environment variables from .env file
load_dotenv()

# Initialize vector store manager with OpenAI embedding function
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize the vector store manager with OpenAI API key
vector_store_manager = VectorStoreManager(openai_api_key=OPENAI_API_KEY)

# Initialize the GamesLoaderService with the vector store manager
games_loader_service = GamesLoaderService(vector_store_manager=vector_store_manager)

# Load the game data from the JSON file into the vector store
vector_store = games_loader_service.load_json(store_name="game_data", json_path="../games.json")

# Query the vector store for a specific game
query = "What is the release date of Hogwarts Legacy?"
results = vector_store.query(query_texts=[query], n_results=1)

# Print the results of the query
print(f"Query: {query}\n")
for result in results['metadatas']:
    for doc in result:
        print(f"-----------Begin Document-----------")
        print(f"Title: {doc['title']}")
        print(f"Developer: {doc['developer']}")
        print(f"Release Date: {doc['release_date']}")
        print(f"Platforms: {doc['platforms']}")
        print(f"Genre: {doc['genre']}")
        print(f"-----------End Document-----------\n")
