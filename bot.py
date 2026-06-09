from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import uvicorn

with open("data.json", "r") as f:
    data = json.load(f)

questions = []
answers = []
answers_table = []

for index, item in enumerate(data):
    answers.append(item["answers"])
    for question in item["questions"]:
        questions.append(question)
        answers_table.append(index)

embedding_model = SentenceTransformer('all-mpnet-base-v2')

cached_embedding_filename = "embeddings.npy"

if not os.path.isfile(cached_embedding_filename):
    print("------------------Generating embeddings...------------------")
    
    embeddings = embedding_model.encode(questions, normalize_embeddings=True)
    np.save(cached_embedding_filename, embeddings)
question_embeddings = np.load(cached_embedding_filename)

def ask_chatbot(question):
  query_embedding = embedding_model.encode([question], normalize_embeddings=True)

  similarities = cosine_similarity(query_embedding, question_embeddings)
  best_match_index = np.argmax(similarities)

  confidence = similarities[0][best_match_index]
  
  if confidence < 0.4:
    return {
        "answer": "The assistant didn't understood the question",
        "confidence": float(confidence)
    }

  return {
    "answer": answers[answers_table[best_match_index]],
    "confidence": float(confidence)
  }