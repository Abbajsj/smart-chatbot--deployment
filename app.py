from flask import Flask, render_template, request, jsonify
from search_engine import SemanticSearchEngine
from logger import log_chat
import os

# ------------------- APP SETUP -------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates")


# ------------------- LOAD KNOWLEDGE -------------------

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


qa_pairs = load_knowledge()

all_questions = []
question_to_answer = []

for questions, answer in qa_pairs:
    for q in questions:
        all_questions.append(q)
        question_to_answer.append(answer)

# ------------------- SEARCH ENGINE -------------------

engine = SemanticSearchEngine(all_questions)


# ------------------- ROUTES -------------------

@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    username = data.get("username", "Guest")
    user_input = data.get("question", "").strip()

    if not user_input:
        return jsonify({
            "answer": "Please enter a valid question.",
            "confidence": 0.0
        })

    # 🔍 Semantic search
    matched_questions = engine.search(user_input)

    if not matched_questions:
        answer = (
            "Sorry, I couldn't find relevant information for your question.\n\n"
            "You can ask about:\n"
            "- 3D printing\n"
            "- Vacuum casting\n"
            "- CAD services\n"
            "- Engineering & manufacturing\n"
            "- Materials\n"
            "- Shipping across Australia\n\n"
            "For custom enquiries, contact info@zeal3dprinting.com.au"
        )
        confidence = 0.0
    else:
        answers = []
        seen = set()

        for mq in matched_questions:
            idx = all_questions.index(mq)
            ans = question_to_answer[idx]

            if ans not in seen:
                answers.append(ans)
                seen.add(ans)

        answer = " ".join(answers)
        confidence = 1.0

    log_chat(username, user_input, answer, confidence)

    return jsonify({
        "answer": answer,
        "confidence": confidence
    })


# ------------------- RUN -------------------

if __name__ == "__main__":
    app.run(debug=True)