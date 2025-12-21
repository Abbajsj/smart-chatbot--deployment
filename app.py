from flask import Flask, render_template, request, jsonify
from search_engine import TextSearchEngine
from logger import log_chat
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder="templates")


def load_knowledge():
    qa_pairs = []
    path = os.path.join(BASE_DIR, "data", "knowledge.txt")

    with open(path, encoding="utf-8") as f:
        questions = []
        answer = None

        for line in f:
            line = line.strip()

            if not line:
                if questions and answer:
                    qa_pairs.append((questions, answer))
                questions = []
                answer = None
                continue

            if line.startswith("Q:"):
                questions.append(line[2:].strip())

            elif line.startswith("A:"):
                answer = line[2:].strip()

        if questions and answer:
            qa_pairs.append((questions, answer))

    return qa_pairs


# -------- Load knowledge --------

qa_pairs = load_knowledge()

all_questions = []
question_to_answer = []

for questions, answer in qa_pairs:
    for q in questions:
        all_questions.append(q)
        question_to_answer.append(answer)

engine = TextSearchEngine(all_questions)


# -------- Routes --------

@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    username = data.get("username", "Guest")
    user_input = data.get("question", "").strip()

    if not user_input:
        return jsonify({
            "answer": "Please enter a question.",
            "confidence": 0.0
        })

    matched_questions, confidence = engine.search_multiple(user_input)

    # No matches found
    if not matched_questions:
        answer = "Sorry, I couldn't find relevant information contact to info@zeal3dprinting.com.au"
    else:
        answers = []
        seen = set()

        for mq in matched_questions:
            idx = engine.documents.index(mq)
            ans = question_to_answer[idx]

            if ans not in seen:
                answers.append(ans)
                seen.add(ans)

        answer = "\n".join(
            f"{i+1}. {a}" for i, a in enumerate(answers)
        )

    log_chat(username, user_input, answer, confidence)

    return jsonify({
        "answer": answer,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)