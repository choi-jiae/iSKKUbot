from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
# from langchain_community.chat_models import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_teddynote.messages import stream_response
# from contextlib import redirect_stdout

# 주의사항: Ollama set up and pull EEVE model 필수! 서버에서 할게요

# Step 1: Set up Pinecone index and embedding model
pinecone_api_key = 'PINCONE_API_KEY'
hf_api_token = "HUGGINGFACE_API_TOKEN"
pinecone_namespace = '2024 1 SKKU  international student handbook.pdf'
# pinecone_namespace = '   Academics  Grading    Course evaluation.txt'

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('iskku-data')

# Define the embeddings model
model_name = "intfloat/multilingual-e5-large-instruct"
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=hf_api_token,
)

# Step 2: Retriever function to extract top_k matches from Pinecone
def retrieve(query, top_k=5):
    embedded_query = hf_embeddings.embed_query(query) # 쿼리로 변환해서 넘기는 것임!
    results = index.query(vector=embedded_query, top_k=top_k, namespace=pinecone_namespace, include_metadata=True)
    return results

import time

query = 'How can I use the shuttle bus to go to the university'
start_time = time.perf_counter()
matches = retrieve(query)
end_time = time.perf_counter()
print(matches)
print(f'Time spent: {end_time-start_time}')

# index_stats = index.describe_index_stats()
# print(index_stats)