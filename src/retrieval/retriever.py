from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re



class Retriever:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="db",
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

        if filter_year:

            print(f"\nFiltering by academic year: {filter_year}")

            results = self.vectorstore.similarity_search(
                query,
                k=15,
                filter={"academic_year": filter_year}
            )

        else:

            results = self.vectorstore.similarity_search(
                query,
                k=15
            )

        print(f"\nResults found: {len(results)}")

        for i, doc in enumerate(results, start=1):

            print(f"\n----- RESULT {i} -----")
            print("Academic Year:", doc.metadata.get("academic_year", "Not Found"))
            print("Source:", doc.metadata.get("source", "Unknown"))
            print("Page:", doc.metadata.get("page", "Unknown"))

        return results