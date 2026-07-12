import time

from src.retrieval.retriever import Retriever
from src.agent.rag_agent import RAGAgent


test_questions = [
    "What are the entry requirements for university admission?",
    "What are the basic qualifications required for Computer Science?",
    "How do I apply for an exam deferral?",
    "What are the library regulations?",
    "What are the academic rules for students?"
]


def evaluate():

    print("\n📊 University RAG Chatbot Evaluation Report")
    print("=" * 50)

    retriever = Retriever()
    agent = RAGAgent()

    total_questions = len(test_questions)

    answered = 0
    unanswered = 0
    total_time = 0


    for i, question in enumerate(test_questions, start=1):

        print(f"\nQuestion {i}: {question}")

        start_time = time.time()

        try:

            results = retriever.get_relevant_chunks(question)

            if results:

                context = "\n\n".join(
                    doc.page_content for doc in results
                )

                answer = agent.generate_answer(
                    question,
                    context
                )


                if isinstance(answer, dict):
                    answer_text = answer.get("answer", "")
                else:
                    answer_text = str(answer)


                if answer_text.strip():

                    answered += 1

                    print("✅ Answer Found")

                else:

                    unanswered += 1

                    print("❌ No Answer")


            else:

                unanswered += 1

                print("❌ No Relevant Documents")


        except Exception as e:

            unanswered += 1

            print("Error:", e)


        end_time = time.time()

        response_time = end_time - start_time

        total_time += response_time

        print(f"⏱ Time: {response_time:.2f}s")


    accuracy = (answered / total_questions) * 100

    avg_time = total_time / total_questions


    print("\n")
    print("=" * 50)
    print("FINAL EVALUATION RESULT")
    print("=" * 50)

    print(f"Total Questions : {total_questions}")
    print(f"Answered        : {answered}")
    print(f"Unanswered      : {unanswered}")
    print(f"Accuracy        : {accuracy:.2f}%")
    print(f"Average Time    : {avg_time:.2f} seconds")

    print("=" * 50)



if __name__ == "__main__":
    evaluate()