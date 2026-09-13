"""Flask server for the NSE strategy signal dashboard.

Run locally:   python webapp/app.py       -> http://127.0.0.1:5000
Deploy:        cd webapp && gunicorn -w 1 -b 0.0.0.0:8000 app:app
               (use ONE worker: signals are cached in process memory)
"""
import os
import sys

from flask import Flask, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import engine  # noqa: E402

app = Flask(__name__)


@app.route("/")
def index():
    engine.start_refresh()          # warm the cache on first hit
    return render_template("index.html")


@app.route("/api/signals")
def api_signals():
    return jsonify(engine.snapshot())


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    started = engine.start_refresh(force=request.args.get("force") == "1")
    return jsonify({"started": started, **engine.snapshot()})


if __name__ == "__main__":
    engine.start_refresh()
    print("\n  Dashboard: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
