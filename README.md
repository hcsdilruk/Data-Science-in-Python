# 🎓 University RAG Chatbot using Python, Ollama & ChromaDB

An AI-powered University Question Answering System built using **Retrieval-Augmented Generation (RAG)**. This chatbot answers student questions by retrieving information from multiple University Student Handbook PDFs and generating accurate responses using **Ollama (Llama 3.2)**.

---

## 📌 Project Overview

This project allows users to ask natural language questions related to university admissions, degree programs, Z-scores, eligibility requirements, and other academic information.

Instead of relying on the language model's knowledge, the chatbot retrieves relevant information directly from the uploaded PDF documents and generates answers based only on those documents.

---

## ✨ Features

- 📄 Read multiple University Handbook PDFs
- ✂️ Split documents into text chunks
- 🧠 Generate embeddings using Sentence Transformers
- 💾 Store embeddings in Chroma Vector Database
- 🔍 Semantic search for relevant document chunks
- 🤖 Generate answers using Ollama (Llama 3.2)
- 📅 Filter documents by Academic Year
- 📚 Supports multiple handbook versions
- 🔁 Interactive question-answer loop

---

# 🏗️ Project Structure

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
│   └── vectorstore/
│         └── chroma_db.py
│
├── ingest.py
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

- Python
- LangChain
- Ollama
- Llama 3.2
- ChromaDB
- Sentence Transformers
- HuggingFace Embeddings
- PyPDF
- Recursive Character Text Splitter

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

Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

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

Download:

https://ollama.com/download

Verify installation

```bash
ollama --version
```

---

## 5. Download Llama 3.2

```bash
ollama pull llama3.2
```

---

## 6. Build the Vector Database

```bash
python ingest.py
```

---

## 7. Run the Chatbot

```bash
python main.py
```

---

# 💬 Example Questions

### Admission

- What are the admission requirements for Medicine in 2022?
- What are the admission requirements for Physiotherapy?
- Which universities offer Biomedical Technology?

---

### Academic Information

- What is the duration of the Medicine degree?
- What is the course code for Physiotherapy?
- Which degree programmes are available in Agriculture?

---

### University Information

- Which university offers Radiography?
- What is the proposed intake for Medicine?
- Which universities offer Engineering?

---

### Z-Score

- What is the minimum Z-score for Medicine in 2023?
- What is the minimum Z-score for Dental Surgery?

---

# 🧠 How It Works

```
PDF Documents
      │
      ▼
PyPDF Loader
      │
      ▼
Text Chunking
      │
      ▼
Sentence Transformer Embeddings
      │
      ▼
Chroma Vector Database
      │
      ▼
Semantic Search
      │
      ▼
Relevant Chunks
      │
      ▼
Ollama (Llama 3.2)
      │
      ▼
Generated Answer
```

---

# 📂 Supported Academic Years

- 2020–2021
- 2022–2023
- 2023–2024
- 2024–2025

The chatbot automatically filters the correct handbook when a year is included in the user's question.

Example:

```
What are the admission requirements for Medicine in 2022?
```

The system retrieves information only from the **2022–2023** handbook.

---

# 📸 Sample Output

```
Ask a Question:

Physiotherapy admission requirements in 2022

Filtering by academic year: 2022-2023

Answer:

The minimum eligibility requirements are:

• Three 'S' passes in Physics, Chemistry and Biology.
• Ordinary Pass (S) in English.
• Candidates must submit certified G.C.E O/L certificate.
```

---

# 🎯 Future Improvements

- Web Interface using Streamlit
- Voice Input
- OCR Support
- Image-based Questions
- Citation-based Answers
- Conversation Memory
- Hybrid Search (Keyword + Semantic)
- PDF Upload from UI
- Multiple LLM Support
- REST API

---

Faculty of Computing

---

# 📄 License

This project is developed for academic and educational purposes.

---
