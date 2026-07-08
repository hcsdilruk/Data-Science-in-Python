from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent
from src.agent.router import QuestionRouter
from src.prediction.predictor import CutoffPredictor


def main():

    print("🎓 Welcome to the University RAG Chatbot!")

    retriever = Retriever()
    agent = RAGAgent()
    predictor = CutoffPredictor()
    router = QuestionRouter(predictor)

    while True:

        question = input(" Ask a Question❓").strip()

        if question.lower() in ["exit", "quit"]:
            print("\n👋 Thank you for using the University RAG Chatbot! \n😊 Goodbye!")
            break

        if router.classify(question) == "prediction":
            print("\n📈 ===== PREDICTION (Historical Cut-off Data) =====\n")
            print(router.answer(question))
        else:
            results = retriever.get_relevant_chunks(question)

            context = "\n\n".join(
                [doc.page_content for doc in results]
            )

            print("\n📄 ===== RETRIEVED CHUNKS =====\n")
            print(context[:1500])

            answer = agent.generate_answer(question, context)

            print("\n💬yANSWERS\n")
            print(answer)

        choice = input("\n🤔 Do you have another question? (yes/no): ").strip().lower()

        if choice not in ["yes", "y"]:
            print("\n🙏 Thank you for using the University RAG Chatbot!")
            print("👋 Have a great day!")
            break


if __name__ == "__main__":
    main()