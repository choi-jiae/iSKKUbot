import os
import warnings
from dotenv import load_dotenv
from pathlib import Path

from pinecone import *
# warnings.filterwarnings("ignore")

load_dotenv()  # Load environment variables from .env file

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings


def embedding_doc(chunk, model_name, hf_embeddings):

    # hf_embeddings = HuggingFaceEndpointEmbeddings(
    #     model=model_name,
    #     task="feature-extraction",
    #     huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
    # )

    # LIMIT = 30
    # if len(chunks)>30:
    #    embedded_documents = []
    #    for i in range(0, len(chunks)+1, 30):
    #         embedded_doc_part = hf_embeddings.embed_documents(chunks[i:min(i+30, len(chunks))])
    #         embedded_documents += embedded_doc_part
    # else:
        # embedded_documents = hf_embeddings.embed_documents(chunks)

    embedded_documents = hf_embeddings.embed_documents(chunk)

    return embedded_documents, hf_embeddings




class PineconeStore:
    def __init__(self, index_name):
        # Load API key from environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Create a Pinecone instance
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Define the index name
        # self.index_name = "iskku-data"
        self.index_name = index_name

        # Check if the index exists, if not create it
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

    def save_vectors(self, vectors, title, content, link):
        # Get the index
        index = self.pc.Index(self.index_name)

        # Iterate over the embeddings and save each one with unique metadata
        for i, vector in enumerate(vectors):
            chunk_metadata = {
                "id": title,
                "text": content,
                "link": link
            }
            # Upsert each vector with its corresponding metadata
            index.upsert(
                vectors=[(title, vector, chunk_metadata)],
            )
                # namespace=os.path.basename(file_path))


