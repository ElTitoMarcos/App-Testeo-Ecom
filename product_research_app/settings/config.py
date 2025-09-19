import os

SSE_ENABLED = os.getenv("SSE_ENABLED", "0") in ("1", "true", "True", "yes")
