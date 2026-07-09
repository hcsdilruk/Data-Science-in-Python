import csv
import os
from datetime import datetime


class UnansweredLogger:

    def __init__(self):
        self.file = "logs/unanswered_questions.csv"

        os.makedirs("logs", exist_ok=True)

        if not os.path.exists(self.file):
            with open(self.file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Question", "Times Asked", "Status"])

    def log_question(self, question):

        rows = []

        found = False

        if os.path.exists(self.file):
            with open(self.file, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        header = rows[0]
        data = rows[1:]

        for row in data:
            if row[1].strip().lower() == question.lower():
                row[2] = str(int(row[2]) + 1)
                found = True
                break

        if not found:
            data.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                question,
                "1",
                "Pending"
            ])

        with open(self.file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)