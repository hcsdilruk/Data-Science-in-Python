from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class Retriever:

    def get_relevant_chunks(self, query):

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            persist_directory="db",
            embedding_function=embedding_model
        )

        results = vectorstore.similarity_search(
            query,
            k=3
        )

        return results