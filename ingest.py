from src.ingestion.pdf_loader import PDFLoader
from src.chunking.chunker import TextChunker
from src.vectorstore.chroma_db import ChromaDBManager
import os
from dotenv import load_dotenv

load_dotenv()

loader = PDFLoader()
chunker = TextChunker()
db_manager = ChromaDBManager()

all_chunks = []

pdf_folder = os.getenv("PDF_DIRECTORY")


for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, file)

        docs = loader.load_pdf(pdf_path)

        year = file.split("(")[1].split(")")[0]

        for doc in docs:
            doc.metadata["academic_year"] = year

        chunks = chunker.split_documents(docs)

        all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")

db_manager.create_vector_store(all_chunks)

print("Vector DB created successfully")
