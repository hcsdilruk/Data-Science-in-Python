import csv
import os
from datetime import datetime


class UnansweredLogger:

    def __init__(self):
        self.file = "logs/unanswered_questions.csv"

        os.makedirs("logs", exist_ok=True)

        self.header = ["Date", "Question", "Times Asked", "Status"]

        # Create file if it doesn't exist or is empty
        if (not os.path.exists(self.file)) or os.path.getsize(self.file) == 0:
            with open(self.file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log_question(self, question):

        rows = []
        found = False

        # Read existing data
        if os.path.exists(self.file):
            with open(self.file, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        # If file has wrong header, recreate it
        if not rows or rows[0] != self.header:
            rows = [self.header]

        data = rows[1:]

        # Check if question already exists
        for row in data:

            # Skip invalid rows
            if len(row) < 4:
                continue

            if row[1].strip().lower() == question.strip().lower():
                row[2] = str(int(row[2]) + 1)
                found = True
                break

        # Add new question
        if not found:
            data.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                question,
                "1",
                "Pending"
            ])

        # Save file
        with open(self.file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(data)

        print(f"✅ Logged unanswered question: {question}")