from src.ingestion.pdf_loader import PDFLoader
from src.chunking.chunker import TextChunker
from src.vectorstore.chroma_db import ChromaDBManager
import os
import re
from dotenv import load_dotenv

load_dotenv()

loader = PDFLoader()
chunker = TextChunker()
db_manager = ChromaDBManager()

all_chunks = []

pdf_folder = os.getenv("PDF_DIRECTORY", "data/pdfs")

for file in os.listdir(pdf_folder):

    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_folder, file)

    docs = loader.load_pdf(pdf_path)

    # Extract academic year from filename
    year_match = re.search(r"\((\d{4}-\d{4})\)", file)

    academic_year = year_match.group(1) if year_match else "Unknown"

    # Add academic year metadata
    for doc in docs:
        doc.metadata["academic_year"] = academic_year

    chunks = chunker.split_documents(docs)

    for chunk in chunks:

        text = chunk.page_content

        # -------------------------
        # Course Code
        # -------------------------
        code_match = re.search(
            r"Course\s*Code\s*[-–:]\s*(\d{3})",
            text,
            re.IGNORECASE
        )

        if code_match:
            chunk.metadata["course_code"] = code_match.group(1)

        # -------------------------
        # Course Name
        # -------------------------
        name_match = re.search(
            r"\d+\.\d+\.\d+\.\d+\s*(.+)",
            text
        )

        if name_match:

            course_name = name_match.group(1).strip()

            # Remove "(Course Code ...)" if it exists
            course_name = re.sub(
                r"\(Course\s*Code.*",
                "",
                course_name
            ).strip()

            chunk.metadata["course_name"] = course_name

    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")

db_manager.create_vector_store(all_chunks)

print("✅ Vector DB created successfully")