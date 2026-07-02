from langchain_ollama import OllamaLLM


class RAGAgent:

    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")

    def generate_answer(self, question, context):

        prompt = f"""
You are a university assistant.

Rules:
1. Use ONLY the provided context.
2. If information is missing, do not guess.
3. Do not use outside knowledge.
4. Quote details directly from the context when possible.
5. If the answer is not present, say:
"I could not find that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response