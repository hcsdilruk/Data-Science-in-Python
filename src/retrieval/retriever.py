from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
import os
from dotenv import load_dotenv

load_dotenv()


class Retriever:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL")
        )

        self.vectorstore = Chroma(
            persist_directory=os.getenv("CHROMA_DB_PATH"),
            embedding_function=self.embedding_model
        )

    def get_relevant_chunks(self, query):

        year_match = re.search(r"20\d{2}", query)

        filter_year = None

        if year_match:

            year = year_match.group()

            year_mapping = {
                "2020": "2020-2021",
                "2022": "2022-2023",
                "2023": "2023-2024",
                "2024": "2024-2025"
            }

            filter_year = year_mapping.get(year)

        try:

            if filter_year:

                results = self.vectorstore.similarity_search(
                    query,
                    k=int(os.getenv("TOP_K")),
                    filter={"academic_year": filter_year}
                )

            else:

                results = self.vectorstore.similarity_search(
                    query,
                    k=int(os.getenv("TOP_K"))
                )

            return results

        except Exception as e:

            print(f"Retrieval Error: {e}")
            return []