from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

with open("data.json", "r") as f:
    data = json.load(f)

questions = [item["questions"] for item in data]
answers = [item["answers"] for item in data]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

cached_embedding_filename = "embeddings.npy"

if not os.path.isfile(cached_embedding_filename):
    print("------------------Generating embeddings...------------------")
    embeddings = embedding_model.encode(questions)
    np.save(cached_embedding_filename, embeddings)
question_embeddings = np.load(cached_embedding_filename)

def ask_chatbot(question):
  query_embedding = embedding_model.encode([question])

  similarities = cosine_similarity(query_embedding, question_embeddings)
  best_match_index = np.argmax(similarities)

  confidence = similarities[0][best_match_index]
  
  if confidence < 0.6:
    return {
        "answer": "I dont understand the question :(",
        "confidence": confidence
    }

  return {
    "answer": answers[best_match_index],
    "confidence": float(confidence)
  }