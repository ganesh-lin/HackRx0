"""
Script to set up Pinecone index for the hackrx project.
Run this script once to create the required index.
"""

from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_pinecone_index():
    """Create Pinecone index if it doesn't exist."""
    try:
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            print("Error: PINECONE_API_KEY not found in environment variables")
            print("Please set PINECONE_API_KEY in your .env file")
            return False
        
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "developer-quickstart-py"
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists!")
            return True
        
        # Create index with appropriate dimensions
        # all-MiniLM-L6-v2 model produces 384-dimensional embeddings
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension for all-MiniLM-L6-v2 embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # You can change this to your preferred region
            )
        )
        
        print(f"Index '{index_name}' created successfully!")
        return True
        
    except Exception as e:
        print(f"Error setting up Pinecone index: {e}")
        return False

if __name__ == "__main__":
    setup_pinecone_index()
