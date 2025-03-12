from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from pinecone import Pinecone
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from dbupdate import dbupdate
from datetime import datetime
import json
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi_utils.tasks import repeat_every  # FastAPI 전용 유틸리티
from pytz import timezone

load_dotenv() 

# Step 1: Set up Pinecone index and embedding model
pinecone_api_key = os.getenv("PINECONE_API_KEY")
hf_api_token = os.getenv("HF_API_TOKEN")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('iskku-data3')
llm = ChatOllama(model='hf.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF:Q4_K_M', temperature=0.1)

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
def retrieve(query, top_k=10):
    embedded_query = hf_embeddings.embed_query(query)
    results = index.query(
        vector=embedded_query,
        top_k=top_k,
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
def generate_response(query, context, matches):
    global llm

    # Use the fixed prompt template for the chat
    prompt = ChatPromptTemplate.from_template("""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.
Human: Use given context, answer the following question in detail as much as you can. This context is about SungKyunKwan University's information (성균관대학교). Write based on user's language. Context: {context} \n Question: {query}
Assistant:\n""")

    # Create the chain
    chain = prompt | llm | StrOutputParser()

    # Generate the response using stream()
    response_iterator = chain.stream({"query": query, "context": context})

    # Generate each response token and append source information at the end in a non-blocking way
    for response in response_iterator:
        yield response
    source_info = "\n\n sources: \n" + "\n".join([
        f"<a href=\"{match.get('metadata', {}).get('link')}\" target=\"_blank\">{match.get('metadata', {}).get('link')}</a>"
        if match.get('metadata', {}).get('link')
        else match.get('metadata', {}).get('source', 'unknown')
        for match in matches
    ])
    yield f"\n{source_info}"

# Database update 함수 호출
def scheduled_db_update():
    try:
        dbupdate()  # 데이터베이스 업데이트 실행
        current_time = datetime.now(timezone("Asia/Seoul")).strftime("%y/%m/%d %H:%M GMT+9")
        with open("update_time.json", "w") as file:
            json.dump({"time": current_time}, file)
        print(f"Database updated successfully at {current_time}")
    except Exception as e:
        print(f"Database update failed: {e}")

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

class ChatRequest(BaseModel):
    chat: str

# FastAPI 애플리케이션 시작 시 스케줄러 설정
@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    # 매일 00:00 (Asia/Seoul 시간대 기준) 업데이트 스케줄링
    scheduler.add_job(scheduled_db_update, "cron", hour=20, minute=00, timezone="Asia/Seoul")
    scheduler.start()

    # 종료 이벤트를 처리해 스케줄러를 안전하게 종료
    @app.on_event("shutdown")
    def stop_scheduler():
        scheduler.shutdown()

@app.post("/chat")
def chat_response(request: ChatRequest):
    # Retrieve relevant contexts from Pinecone
    matches = retrieve(request.chat)
    
    # Rerank the retrieved contexts
    reranked_matches = rerank(request.chat, matches)
    retrieved_context = "\n".join([match['metadata']['text'] for match in reranked_matches[:3]])

    # Generate the response iterator
    response_iterator = generate_response(request.chat, retrieved_context, reranked_matches[:3])

    # Return the StreamingResponse directly
    return StreamingResponse(response_iterator, media_type="text/html")

@app.get("/update_db")
def update_db():
    dbupdate()  # Call the database update function
    
    # Get the current time in the desired format
    current_time = datetime.now(timezone("Asia/Seoul")).strftime("%y/%m/%d %H:%M GMT+9")
    
    # Write the current time to a file
    with open("update_time.json", "w") as file:
        json.dump({"time": current_time}, file)
    
    return {"message": "Database updated successfully."}

# Retrieve the last update time
@app.get("/last_update")
def last_update():
    try:
        # Read the time from the file
        with open("update_time.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        # Handle the case where the file does not exist
        return {"error": "Update time not found. Please run /update_db first."}
    
    # Return the time in the desired JSON format
    return {"time": data["time"]}
