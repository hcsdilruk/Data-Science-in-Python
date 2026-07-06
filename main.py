from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent
from src.agent.router import QuestionRouter
from src.prediction.predictor import CutoffPredictor


def main():

    retriever = Retriever()
    agent = RAGAgent()
    predictor = CutoffPredictor()
    router = QuestionRouter(predictor)

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if router.classify(question) == "prediction":
            print("\n===== PREDICTION (from historical cut-off data) =====\n")
            print(router.answer(question))
            continue

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
