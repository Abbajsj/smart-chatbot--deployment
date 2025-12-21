import csv
from datetime import datetime

FILE_PATH = "data/chat_history.csv"


def log_chat(username, question, answer, confidence):
    with open(FILE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            username,
            question,
            answer,
            round(confidence, 3)
        ])