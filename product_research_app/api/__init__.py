from flask import Flask

app = Flask(__name__)

from . import config  # noqa: E402,F401
