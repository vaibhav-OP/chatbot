# Simple Semantic Q&A Chatbot

This is a simple semantic Q&A chatbot built using FastAPI and Sentence Transformers.

Instead of matching exact keywords, the chatbot converts questions into embeddings and compares semantic similarity using cosine similarity.

The project also caches embeddings locally to avoid regenerating vectors every time the server starts.

## Features

* Semantic question matching
* Sentence embeddings using `all-MiniLM-L6-v2`
* FastAPI REST API
* Embedding caching with NumPy
* Cosine similarity search
* Confidence thresholding


## Requirements

Make sure these are installed on your machine:

* Python 3.10 or newer
* pip

Check installation:

```bash
python --version
pip --version
```

## Clone the Repository

```bash
git clone <your-repository-url>
cd <project-folder>
```

## Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will run locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Example Request

```http
GET /chat?question=I want my money back
```

Example response:

```json
{
  "answer": "You can request a refund from the settings page.",
  "confidence": 0.82
}
```

## How Embedding Caching Works

The first time the server starts:

1. Questions from `data.json` are converted into embeddings.
2. The embeddings are saved into `embeddings.npy`.

On future runs:

* Cached embeddings are loaded directly from disk.
* This avoids expensive embedding generation every startup.

## Updating Questions

If you update `data.json`:

1. Delete `embeddings.npy`
2. Restart the server

New embeddings will automatically generate.

## Technologies Used

* Python
* FastAPI
* Sentence Transformers
* scikit-learn
* NumPy

## Notes

This is a retrieval-based chatbot, not a generative AI model.

It retrieves the closest known answer using semantic similarity between embeddings.
