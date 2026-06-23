from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent


def main():

    question = input("Ask a Question: ")

    retriever = Retriever()

    results = retriever.get_relevant_chunks(question)

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    print("\n===== RETRIEVED CHUNKS =====\n")
    print(context[:1500])

    agent = RAGAgent()

    answer = agent.generate_answer(
        question,
        context
    )

    print("\n===== ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()