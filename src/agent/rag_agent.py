from langchain_ollama import OllamaLLM


class RAGAgent:

    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")

    def generate_answer(self, question, context):

        prompt = f"""
You are a university assistant.

Use ONLY the provided context to answer.

If the answer is not in the context, say:
"I could not find that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response