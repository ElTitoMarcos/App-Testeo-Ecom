from flask import Flask, jsonify, request
from server.logging_setup import setup_logging
from analysis.analyzer import analyze

LOG_PATH = setup_logging()

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_error(e):
  app.logger.exception("Unhandled")
  return jsonify(ok=False, message="Ha ocurrido un error", log_path=LOG_PATH), 500

@app.get('/api/log-path')
def log_path():
  return jsonify(ok=True, log_path=LOG_PATH)


@app.post('/api/analyze')
def api_analyze():
  data = request.get_json(force=True)
  return jsonify(ok=True, result=analyze(data))
