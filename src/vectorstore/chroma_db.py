from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


class ChromaDBManager:

    def __init__(self):

        self.persist_directory = os.getenv(
            "CHROMA_DB_PATH",
            "db/chroma"
        )

        os.makedirs(
            self.persist_directory,
            exist_ok=True
        )


    def create_vector_store(self, chunks):

        print("Creating embeddings...")

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=self.persist_directory
        )


        print(
            f"Vector DB saved at: {self.persist_directory}"
        )

        return vectorstore