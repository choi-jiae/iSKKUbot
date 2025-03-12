import pdfplumber
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

import glob
import pandas as pd
from pinecone import *
import warnings
from dotenv import load_dotenv
import time

class PineconeStore:
    def __init__(self):
        # Load API key from environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Create a Pinecone instance
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Define the index name
        self.index_name = "iskku-data3"

    def save_vectors(self, vectors, metadata, chunks):
        # Get the index
        index = self.pc.Index(self.index_name)

        # Iterate over the embeddings and save each one with unique metadata
        for i, vector in enumerate(vectors):
            # ignore empty chunk
            if (chunks[i] != ''):
                vector_id = f"{os.path.basename(metadata['source'])}_page{i}"  # Unique ID for each chunk
                chunk_metadata = {
                    "id": vector_id,
                    "source": vector_id,
                    "chunk": i,
                    "text": chunks[i]  # Add the text of the chunk here
                }
                # Upsert each vector with its corresponding metadata
                index.upsert(
                    vectors=[(vector_id, vector, chunk_metadata)]
                    )

def pdf_metadata_loader(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    pages = loader.load()

    return pages[0].metadata

def df_to_text(df):
    text = ""
    for _, row in df.iterrows():
        sentence = ", ".join([f"{col}: {value}" for col, value in row.items()])
        text += sentence + "\n" 
    return text
    
def extract_text_and_tables(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        extracted_pages = []

        for page in pdf.pages:
            all_tables = ''
            # Extract text
            all_texts = page.extract_text()

            # Extract table
            settings = {
                "horizontal_strategy": "lines",
                "vertical_strategy": "text",
                "intersection_tolerance": 5,
                "snap_tolerance": 4,
                "text_x_tolerance": 7
            }
            
            tables = page.extract_tables(settings)
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                all_tables += df_to_text(df)
            
            # Concat text and table
            extracted_pages.append(all_texts + '\n' + all_tables)
    return extracted_pages

def embedding_doc(chunks, model_name):

    hf_embeddings = HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
    )
    LIMIT = 10
    print('saving chunks...length:', len(chunks))
    if len(chunks)>LIMIT:
       time.sleep(1)
       embedded_documents = []
       for i in range(0, len(chunks)+1, LIMIT):
            print(f'saving {i}~{i+LIMIT-1}page')
            embedded_doc_part = hf_embeddings.embed_documents(chunks[i:min(i+LIMIT, len(chunks))])
            embedded_documents += embedded_doc_part
    else:
        embedded_documents = hf_embeddings.embed_documents(chunks)
    print('Done!')
    return embedded_documents


def main():
    warnings.filterwarnings("ignore")
    load_dotenv()
    file_path = '../../data/pdfs/*.pdf'
    pdf_files = glob.glob(file_path)

    for file in pdf_files:
        metadata = pdf_metadata_loader(file)
        chunks = extract_text_and_tables(file)

        embeddings = embedding_doc(chunks, model_name="intfloat/multilingual-e5-large-instruct")

        vector_store = PineconeStore()
        vector_store.save_vectors(embeddings, metadata, chunks)

if __name__ == "__main__":
    main()