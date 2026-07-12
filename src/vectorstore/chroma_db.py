from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class ChromaDBManager:

    def create_vector_store(self, chunks):

        embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL")
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=os.getenv("CHROMA_DB_PATH")
        )

        return vectorstore