from flask import Flask, request, jsonify
import threading
import queue

# A thread-safe queue so Streamlit can read new signals
incoming_signals = queue.Queue()

app = Flask(__name__)

@app.route("/news", methods=["POST"])
def receive_signal():
    data = request.get_json()

    # Validate minimum fields
    required = ["symbols", "sentiment", "impact", "confidence", "headline", "timestamp"]
    for r in required:
        if r not in data:
            return jsonify({"error": f"Missing field: {r}"}), 400

    # Store signal so Streamlit can use it
    incoming_signals.put(data)

    return jsonify({"status": "received"}), 200

def start_flask():
    app.run(host="0.0.0.0", port=5000)
