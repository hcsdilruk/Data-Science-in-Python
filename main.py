from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent
from src.agent.router import QuestionRouter
from src.prediction.predictor import CutoffPredictor
from src.utils.answered_manager import AnsweredManager
from src.logs.unanswered_logger import UnansweredLogger
from dotenv import load_dotenv
load_dotenv()


def main():

    print("🎓 Welcome to the University RAG Chatbot!")

    retriever = Retriever()
    agent = RAGAgent()
    predictor = CutoffPredictor()
    router = QuestionRouter(predictor)

    answered = AnsweredManager()
    logger = UnansweredLogger()

    while True:

        question = input("\nAsk a Question❓ ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\n👋 Thank you for using the University RAG Chatbot!")
            break

        # ---------- Prediction Questions ----------
        if router.classify(question) == "prediction":
            print("\n💬 ANSWER\n")
            print(router.answer(question))
        else:

            # ---------- PDF Retrieval ----------
            results = retriever.get_relevant_chunks(question)
            print("DEBUG: Number of chunks:", len(results))

            answer_text = ""

            if results:

                print("\n" + "=" * 60)
                print("📄 RETRIEVED CHUNKS")
                print("=" * 60)

                for i, doc in enumerate(results, 1):

                    year = doc.metadata.get("academic_year", "Unknown")

                    print(f"\nChunk {i} | Academic Year: {year}")
                    print("-" * 60)
                    print(doc.page_content[:500])

                print("\n" + "=" * 60)

                context = "\n\n".join(
                    doc.page_content
                    for doc in results[:3]
                )
                answer = agent.generate_answer(question, context)

                if isinstance(answer, dict):
                    found = answer.get("found", False)
                    answer_text = answer.get("answer", "").strip()
                else:
                    found = True
                    answer_text = str(answer).strip()

                # If LLM couldn't answer
                if (
                    not found
                    or answer_text == ""
                    or answer_text.lower().startswith("i could not find")
                ):
                    answer_text = ""

            # ---------- Check answered_questions.csv ----------
            if not answer_text:

                print("DEBUG: answer_text is empty")

                csv_answer = answered.get_answer(question)

                if csv_answer:
                    print("DEBUG: Found answer in answered_questions.csv")
                    answer_text = csv_answer
                else:
                    print("DEBUG: Logging unanswered question...")
                    logger.log_question(question)
                    print("DEBUG: Logged successfully")
                    answer_text = (
                        "I could not find that information in the provided documents."
                    )

            # ---------- Display Final Answer ----------
            print("\n💬 ANSWER\n")
            print(answer_text)

        again = input("\nAnother question? (yes/no): ").strip().lower()

        if again not in ["yes", "y"]:
            print("\n👋 Thank you for using the University RAG Chatbot!")
            break


if __name__ == "__main__":
    main()
