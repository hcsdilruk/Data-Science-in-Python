from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent


def main():

    retriever = Retriever()
    agent = RAGAgent()

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        results = retriever.get_relevant_chunks(question)

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        print("\n===== RETRIEVED CHUNKS =====\n")
        print(context[:1500])

        answer = agent.generate_answer(
            question,
            context
        )

        print("\n===== ANSWER =====\n")
        print(answer)


if __name__ == "__main__":
    main()