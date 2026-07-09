# 🎓 University RAG Chatbot using Python, Ollama & ChromaDB

An AI-powered **University Question Answering System** built using **Retrieval-Augmented Generation (RAG)**.

This chatbot answers student questions by retrieving information from official **University Student Handbook PDF documents** and generating accurate responses using **Ollama (Llama 3.2 Large Language Model)**.

The system is designed to provide reliable document-based answers while reducing AI hallucinations by restricting responses to information available in the uploaded university documents.

---

# 📌 Project Overview

Traditional chatbots often depend on predefined answers or general AI knowledge, which can produce inaccurate information.

This project uses **Retrieval-Augmented Generation (RAG)** to solve this problem.

The system first retrieves relevant information from university handbook documents and then uses an AI language model to generate answers based only on the retrieved context.

Users can ask questions related to:

* University admission requirements
* Degree programs
* Z-score requirements
* Eligibility criteria
* Academic regulations
* Course information
* University details

---

# 🔎 What is RAG?

Retrieval-Augmented Generation combines two major processes:

## 1. Retrieval

The system searches the document database and finds the most relevant information related to the user's question.

Example:

Question:

```
What is the minimum mark required for the Common General Paper?
```

Retrieved document information:

```
Candidates who have not obtained 30% or above for the Common General Paper are not eligible...
```

---

## 2. Generation

The retrieved information is provided to the Large Language Model (Llama 3.2), which generates a natural language answer.

The model is instructed to:

* Use only provided document information
* Avoid guessing
* Avoid external knowledge
* Return "information not found" when required

---

# ✨ Features

* 📄 Load multiple University Student Handbook PDFs
* ✂️ Split large documents into manageable text chunks
* 🧠 Generate semantic embeddings using Sentence Transformers
* 💾 Store embeddings using Chroma Vector Database
* 🔍 Perform semantic similarity search
* 🤖 Generate answers using Ollama Llama 3.2
* 📅 Academic year-based document filtering
* 📚 Support multiple handbook versions
* 🚫 Reduce hallucinated answers
* 🔁 Interactive question-answer system
* 📝 Handle unanswered questions

---

# 🏗️ System Architecture

```
              PDF Documents
                    |
                    ▼
              PDF Loader
                    |
                    ▼
              Text Chunking
                    |
                    ▼
              Embedding Model
                    |
                    ▼
              Chroma Vector Database
                    |
                    ▼
              Semantic Retriever
                    |
                    ▼
              Relevant Context
                    |
                    ▼
              Ollama Llama 3.2
                    |
                    ▼
              Final Answer
```

---

# 📂 Project Structure

```
Data-Science-in-Python
│
├── data/
│   └── pdfs/
│       ├── student_handbook_english (2020-2021).pdf
│       ├── student_handbook_english (2022-2023).pdf
│       ├── student_handbook_english (2023-2024).pdf
│       └── student_handbook_english (2024-2025).pdf
│
├── db/
│   └── chroma.sqlite3
│
├── src/
│   ├── agent/
│   │     └── rag_agent.py
│   │
│   ├── chunking/
│   │     └── chunker.py
│   │
│   ├── ingestion/
│   │     └── pdf_loader.py
│   │
│   ├── retrieval/
│   │     └── retriever.py
│   │
│   ├── vectorstore/
│   │     └── chroma_db.py
│   │
│   └── utils/
│         └── answered_manager.py
│
├── ingest.py
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

| Component            | Technology                        |
| -------------------- | --------------------------------- |
| Programming Language | Python                            |
| Framework            | LangChain                         |
| LLM                  | Ollama Llama 3.2                  |
| Vector Database      | ChromaDB                          |
| Embedding Model      | Sentence Transformers             |
| Embedding Library    | HuggingFace Embeddings            |
| PDF Processing       | PyPDF                             |
| Text Splitting       | Recursive Character Text Splitter |

---

# ⚙️ How the System Works

## 1. Document Ingestion

PDF documents are loaded using PyPDF.

Example:

```
student_handbook_2022-2023.pdf
student_handbook_2023-2024.pdf
student_handbook_2024-2025.pdf
```

The extracted text is prepared for processing.

---

## 2. Text Chunking

Large PDF documents are divided into smaller sections.

Configuration:

```
Chunk Size: 500
Chunk Overlap: 100
```

Chunk overlap helps maintain context between sections.

---

## 3. Embedding Generation

Each chunk is converted into a numerical vector using:

```
sentence-transformers/all-MiniLM-L6-v2
```

These vectors represent the semantic meaning of the text.

---

## 4. Vector Storage

All embeddings are stored in:

```
Chroma Vector Database
```

This allows fast similarity-based searching.

---

## 5. Retrieval

When a user asks a question:

1. The question is converted into an embedding.
2. Chroma searches similar document chunks.
3. Top relevant chunks are selected.

Example:

```
similarity_search(query, k=5)
```

---

## 6. Answer Generation

The retrieved context is sent to:

```
Ollama Llama 3.2
```

The model generates an answer using only the retrieved information.

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/Data-Science-in-Python.git

cd Data-Science-in-Python
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download Ollama:

```
https://ollama.com/download
```

Check installation:

```bash
ollama --version
```

---

## 5. Download Llama 3.2 Model

```bash
ollama pull llama3.2
```

---

## 6. Create Vector Database

Run:

```bash
python ingest.py
```

This process:

* Loads PDFs
* Creates chunks
* Generates embeddings
* Stores vectors in ChromaDB

---

## 7. Start Chatbot

```bash
python main.py
```

---

# 💬 Example Questions

### Admission

```
What are the admission requirements for Medicine in 2022?
```

```
What are the admission requirements for Physiotherapy?
```

---

### Academic Information

```
What is the duration of the Medicine degree?
```

```
Which degree programmes are available in Agriculture?
```

---

### Z-Score

```
What is the minimum Z-score for Medicine in 2023?
```

---

### General Requirements

```
What is the minimum mark required for the Common General Paper?
```

---

# 📅 Supported Academic Years

The chatbot currently supports:

* 2020–2021
* 2022–2023
* 2023–2024
* 2024–2025

If a year is mentioned in the question, the system automatically filters the relevant handbook.

Example:

```
Admission requirements for Medicine in 2022
```

Retrieves:

```
student_handbook_2022-2023.pdf
```

---

# 🧪 Sample Output

```
Ask a Question:

What is the minimum mark required for the Common General Paper?


Answer:

The minimum required mark is 30%.

Candidates who have not obtained 30% or above for the Common General Paper
are not eligible for registration for selected university courses.
```

---

# 🔐 Hallucination Prevention

The chatbot reduces incorrect AI answers by:

* Using only retrieved PDF content
* Preventing external knowledge usage
* Returning "information not found" when data is unavailable

---

# 🚧 Challenges Faced

## Document Retrieval

Problem:
Relevant information was not always retrieved.

Solution:
Improved chunking and semantic similarity search.

---

## AI Hallucination

Problem:
LLMs may generate incorrect information.

Solution:
Implemented strict context-based prompting.

---

## PDF Formatting Issues

Problem:
Some PDF text extraction contains formatting errors.

Solution:
Used text chunking and preprocessing.

---

# 🎯 Future Improvements

* Web interface using Streamlit or React
* Voice input support
* OCR support for scanned PDFs
* Image-based questions
* Citation-based answers
* Conversation memory
* Hybrid keyword + semantic search
* PDF upload through UI
* REST API deployment
* Multiple language support

---

# 👨‍🎓 Academic Project

Developed for: DS

**Faculty of Computing**

