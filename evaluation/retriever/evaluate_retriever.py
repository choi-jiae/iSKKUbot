from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv
import argparse
import json
from tqdm import tqdm

load_dotenv('../../.env')

# Step 1: Set up Pinecone index and embedding model
pinecone_api_key = os.getenv("PINECONE_API_KEY")
hf_api_token = os.getenv("HF_API_TOKEN")
pinecone_namespace = 'web'

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('iskku-data3')
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
def retrieve(query, top_k=200):
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


def main(args):
    with open('../eval_dataset_final.json', 'r') as f:
        eval_dataset = json.load(f)

    correct = 0
    for data in tqdm(eval_dataset):
        # data = json.loads(data)
        title = data['title']
        question = data['question']

        # retrieve
        matches = retrieve(question)
        # print(matches[:3])
        if args.rerank:
            matches = rerank(question, matches)
        matches = matches[:args.topk]
        # print(matches)

        matches_titles = []
        for match in matches:
            matches_titles.append(match['id'])
        # print(matches_titles)
        # print(title)
        if title in matches_titles:
            correct += 1
        # print(correct)

    with open(f'./rerank_{args.rerank}_topk_{args.topk}.txt', 'w') as f:
        f.write(f'ACC: {correct/len(eval_dataset):0.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--topk', type=int, required=True)
    args = parser.parse_args()

    main(args)