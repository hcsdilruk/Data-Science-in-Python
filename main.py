from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent
from src.agent.router import QuestionRouter
from src.prediction.predictor import CutoffPredictor
from src.utils.answered_manager import AnsweredManager
from src.logs.unanswered_logger import UnansweredLogger
from dotenv import load_dotenv

load_dotenv()

# Phrases that indicate the LLM could not answer from the handbook
FAILURE_PHRASES = [
    "not mentioned",
    "not provided",
    "not found",
    "no information",
    "cannot answer",
    "i don't know",
    "does not contain",
    "could not find",
    "not available",
    "outside the provided context",
]


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
                    for doc in results[:4]
                )

                answer = agent.generate_answer(question, context)

                if isinstance(answer, dict):
                    found = answer.get("found", False)
                    answer_text = answer.get("answer", "").strip()
                else:
                    found = True
                    answer_text = str(answer).strip()

                # Detect if the LLM failed to answer
                if (
                    not found
                    or answer_text == ""
                    or any(
                        phrase in answer_text.lower()
                        for phrase in FAILURE_PHRASES
                    )
                ):
                    answer_text = ""

            # ---------- Check answered_questions.csv ----------
            if not answer_text:

                csv_answer = answered.get_answer(question)

                if csv_answer:
                    answer_text = csv_answer
                else:
                    logger.log_question(question)
                    answer_text = (
                        "This question is outside the scope of the University Handbook.\n"
                        "Your question has been recorded for future improvements."
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