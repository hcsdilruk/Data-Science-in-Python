from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

load_dotenv()


class RAGAgent:

    def __init__(self):

        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL"),
            temperature=0
        )


    def generate_answer(self, question, context):

        prompt = f"""
You are a University Admission Assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Instructions:

- Read the entire context carefully before answering.
- Answer only using information found in the context.
- If the question is about a specific course, use ONLY that course's information.
- Do not mix information from different courses.
- If the question is about general university admission, application procedures, rules, FAQs, eligibility, or other handbook information, answer from the relevant sections.
- Do not guess or use outside knowledge.
- Keep the answer clear and under 100 words.
- If the answer is not found in the context, reply exactly:

I could not find that information in the provided documents.


Answer:
"""

        response = self.llm.invoke(prompt)

        print("\n========== LLM RESPONSE DEBUG ==========")
        print(response)
        print("========================================")

        response = response.strip()


        if (
            response
            and not response.lower().startswith(
                "i could not find"
            )
        ):
            return {
                "found": True,
                "answer": response
            }


        return {
            "found": False,
            "answer": ""
        }