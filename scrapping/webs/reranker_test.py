from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from pinecone import Pinecone
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Step 1: Set up Pinecone index and embedding model
pinecone_api_key = os.getenv("PINECONE_API_KEY")
hf_api_token = os.getenv("HF_API_TOKEN")
pinecone_namespace = ''

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('iskku-data')
# llm = ChatOllama(model='hf.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF:Q4_K_M', temperature=0.1)

# Define the embeddings model
model_name = "intfloat/multilingual-e5-large-instruct"
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=hf_api_token,
)

# Step 1.5: Define the reranker model
tokenizer = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
reranker_model = AutoModelForSequenceClassification.from_pretrained("Dongjin-kr/ko-reranker")

# Step 2: Retriever function to extract top_k matches from Pinecone
def retrieve(query, top_k=5):
    embedded_query = hf_embeddings.embed_query(query)
    results = index.query(
        vector=embedded_query,
        top_k=top_k,
        namespace=pinecone_namespace,
        include_metadata=True
    )
    return results['matches']

# Step 3: Reranker function to reorder retrieved contexts
def rerank(query, matches):
    reranked_matches = []
    inputs = []
    for match in matches:
        context_text = match['metadata']['text']
        inputs.append((query, context_text))

    # Tokenize and create input tensors
    tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = reranker_model(**tokenized_inputs).logits.squeeze()

    # Attach scores to matches and sort
    for i, match in enumerate(matches):
        match['score'] = scores[i].item()
    reranked_matches = sorted(matches, key=lambda x: x['score'], reverse=True)

    return reranked_matches

# Step 4: Define the generator component using ChatOllama
# def generate_response(query, context):
#     global llm

#     # Use the fixed prompt template for the chat
#     prompt = ChatPromptTemplate.from_template("""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.
# Human: Use given context, answer the following question in detail as much as you can. This context is about SungKyunKwan University's information (성균관대학교). Write based on user's language. Context: {context} \n Question: {query}
# Assistant:\n""")

#     # Create the chain
#     chain = prompt | llm | StrOutputParser()

#     # Generate the response using stream()
#     response_iterator = chain.stream({"query": query, "context": context})

#     return response_iterator

# # Step 5: RAG pipeline combining retriever, reranker, and generator
# app = FastAPI()

# class ChatRequest(BaseModel):
#     chat: str

# @app.post("/chat")
# def chat_response(request: ChatRequest):
#     # Retrieve relevant contexts from Pinecone
#     matches = retrieve(request.chat)
    
#     # Rerank the retrieved contexts
#     reranked_matches = rerank(request.chat, matches)
#     retrieved_context = "\n".join([match['metadata']['text'] for match in reranked_matches])

#     # Generate the response iterator
#     response_iterator = generate_response(request.chat, retrieved_context)

#     # Return the StreamingResponse directly
#     return StreamingResponse(response_iterator, media_type="text/plain")



query = 'What is the process to apply for a part-time job at Sungkyunkwan University?'
matches = retrieve(query)
reranked_matches = rerank(query, matches)

print(matches)
print('==================')
print(reranked_matches)