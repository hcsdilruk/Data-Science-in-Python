from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent
from src.agent.router import QuestionRouter
from src.prediction.predictor import CutoffPredictor
from src.utils.answered_manager import AnsweredManager
from src.logs.unanswered_logger import UnansweredLogger

def main():

    print("🎓 Welcome to the University RAG Chatbot!")

    retriever = Retriever()
    agent = RAGAgent()
    predictor = CutoffPredictor()
    router = QuestionRouter(predictor)

    answered = AnsweredManager()
    logger = UnansweredLogger()

    while True:

        question = input(" Ask a Question❓ ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if router.classify(question) == "prediction":

            print(router.answer(question))
            continue

        # ---------- PDF Search ----------

        results = retriever.get_relevant_chunks(question)

        if results:

            context = "\n\n".join(
                doc.page_content for doc in results
            )

            answer = agent.generate_answer(question, context)

            # If LLM says answer not found, check CSV
            if isinstance(answer, dict):
                answer_text = answer.get("answer", "")
            else:
                answer_text = answer


            if answer_text.lower().startswith("i could not find"):

                csv_answer = answered.get_answer(question)

                if csv_answer:

                    print("\n💬 ANSWER\n")
                    print(csv_answer)

                else:

                    logger.log_question(question)

                    print("\n💬 ANSWER\n")
                    print("I could not find that information in the provided documents.")

            else:

                print("\n💬 ANSWER\n")
                print(answer)

        else:

            # ---------- answered_questions.csv ----------

            csv_answer = answered.get_answer(question)

            if csv_answer:

                print("\n💬 ANSWER\n")
                print(csv_answer)

            else:

                logger.log_question(question)

                print("\n💬 ANSWER\n")
                print("I could not find that information in the provided documents.")

        again = input("\nAnother question? (yes/no): ").lower()

        if again not in ["yes", "y"]:
            break


if __name__ == "__main__":
    main()