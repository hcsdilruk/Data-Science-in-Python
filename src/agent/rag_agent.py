from langchain_ollama import OllamaLLM
import json
import re

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
You are a university document assistant.

Use ONLY the context below.

Context:
{context}

Question:
{question}

Give the answer directly.

If the information exists, answer it.
If not, say:
I could not find that information in the provided documents.

Answer:
"""


        response = self.llm.invoke(prompt)

        response = response.strip()


        if response:
            return {
                "found": True,
                "answer": response
            }


        return {
            "found": False,
            "answer": ""
        }