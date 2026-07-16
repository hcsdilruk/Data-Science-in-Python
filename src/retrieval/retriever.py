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

        course_match = re.search(
            r"(?:course\s*(?:code)?|code)\s*[-:]?\s*(\d{3})",
            query,
            re.IGNORECASE
        )

        filter_dict = {}

        if year_match:

            year = year_match.group()

            year_mapping = {
                "2020": "2020-2021",
                "2022": "2022-2023",
                "2023": "2023-2024",
                "2024": "2024-2025"
            }

            if year in year_mapping:
                filter_dict["academic_year"] = year_mapping[year]

        k = int(os.getenv("TOP_K", 5))

        THRESHOLD = 0.25

        try:

            if filter_dict:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query=query,
                    k=k
                )

            filtered_docs = []

            print("\nDEBUG: Relevance Scores")
            print("-" * 40)

            for doc, score in results:
                print(f"{score:.3f}")

                if score >= THRESHOLD:
                    filtered_docs.append(doc)

            if course_match:

                course_code = course_match.group(1)

                exact = []
                others = []

                for doc in filtered_docs:

                    if re.search(
                        rf"Course\s*Code\s*[-–:]?\s*{course_code}",
                        doc.page_content,
                        re.IGNORECASE
                    ):
                        exact.append(doc)
                    else:
                        others.append(doc)

                if exact:
                    filtered_docs = exact
                else:
                    filtered_docs = others

            return filtered_docs

        except Exception as e:

            print(f"Retrieval Error: {e}")
            return []