import time

from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent


# expected = True  -> System should answer
# expected = False -> System should not answer

test_questions = [
    {
        "question": "What are the entry requirements for university admission?",
        "expected": True,
    },
    {
        "question": "What are the basic qualifications required for Computer Science?",
        "expected": True,
    },
    {
        "question": "How do I apply for an exam deferral?",
        "expected": True,
    },
    {
        "question": "What are the library regulations?",
        "expected": True,
    },
    {
        "question": "What are the academic rules for students?",
        "expected": True,
    },
    {
        "question": "Who is the President of Sri Lanka in 1990?",
        "expected": False,
    },
    {
        "question": "What is the capital city of Japan?",
        "expected": False,
    }
]


def evaluate():

    print("\n📊 University RAG Chatbot Evaluation Report")
    print("=" * 60)

    retriever = Retriever()
    agent = RAGAgent()

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    total_time = 0

    for i, item in enumerate(test_questions, start=1):

        question = item["question"]
        expected = item["expected"]

        print(f"\nQuestion {i}: {question}")

        start = time.time()

        predicted = False

        try:

            results = retriever.get_relevant_chunks(question)

            if results:

                context = "\n\n".join(
                    doc.page_content for doc in results
                )

                answer = agent.generate_answer(question, context)

                if isinstance(answer, dict):
                    answer_text = answer.get("answer", "")
                else:
                    answer_text = str(answer)

                if answer_text.strip():
                    predicted = True

            # -------- Confusion Matrix --------

            if predicted and expected:
                TP += 1
                print("✅ True Positive")

            elif predicted and not expected:
                FP += 1
                print("⚠ False Positive")

            elif not predicted and expected:
                FN += 1
                print("❌ False Negative")

            else:
                TN += 1
                print("✅ True Negative")

        except Exception as e:

            print("Error:", e)

            if expected:
                FN += 1
            else:
                TN += 1

        end = time.time()

        response_time = end - start

        total_time += response_time

        print(f"⏱ Time : {response_time:.2f} seconds")

    total = TP + FP + TN + FN

    accuracy = (TP + TN) / total if total else 0

    precision = TP / (TP + FP) if (TP + FP) else 0

    recall = TP / (TP + FN) if (TP + FN) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    avg_time = total_time / total if total else 0

    print("\n")
    print("=" * 60)
    print("FINAL EVALUATION RESULT")
    print("=" * 60)

    print(f"Total Questions : {total}")
    print(f"TP              : {TP}")
    print(f"FP              : {FP}")
    print(f"TN              : {TN}")
    print(f"FN              : {FN}")

    print("-" * 60)

    print(f"Accuracy        : {accuracy * 100:.2f}%")
    print(f"Precision       : {precision * 100:.2f}%")
    print(f"Recall          : {recall * 100:.2f}%")
    print(f"F1-Score        : {f1 * 100:.2f}%")
    print(f"Average Time    : {avg_time:.2f} seconds")

    print("=" * 60)


if __name__ == "__main__":
    evaluate()