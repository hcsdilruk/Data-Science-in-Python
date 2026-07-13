import csv


class AnsweredManager:

    def __init__(self):

        self.file = "data/answered_questions.csv"

    def get_answer(self, question):

        with open(self.file, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:

                q = row.get("Question") or row.get("question")

                if q and q.strip().lower() == question.strip().lower():

                    return row.get("Answer") or row.get("answer")

        return None