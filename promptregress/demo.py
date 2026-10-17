"""Fake OpenAI-compatible model server.

In-process stand-in for a real model: maps prompt substrings -> canned
responses, served over HTTP on localhost. Used two ways:

  - tests:  from promptregress.demo import start_fake_model
  - --demo: promptregress --demo run ...  (zero-dependency trial)

Maps prompt substrings -> canned responses. Run in a thread:

    server, url = start_fake_model({"hello": "world"})
    ... PROMPTREGRESS_BASE_URL=url ...
    server.shutdown()
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    routes: dict[str, str] = {}
    default = "ok"

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode() or "{}")
        prompt = ""
        for m in body.get("messages", []):
            prompt += m.get("content", "")
        response = self.default
        for key, val in self.routes.items():
            if key in prompt:
                response = val
                break
        payload = json.dumps({
            "choices": [{"message": {"role": "assistant",
                                     "content": response}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def start_fake_model(routes: dict[str, str], default: str = "ok"):
    handler = type("H", (_Handler,), {"routes": routes, "default": default})
    server = HTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, url


# Routes matching the example scaffolded by `promptregress init`
# (first matching key wins). v1 answers every case in one word; v2 — the
# reworded prompt — starts chatting on the neutral review, so `--demo diff
# v1 v2` and `--demo run --version v2` show the regression the README
# describes. The v2 key is the tail of the rendered v2 prompt, so only
# v2 + the neutral review hits it.
CHATTY_NEUTRAL = ("I'd call this one neutral, leaning slightly positive: "
                  "the customer seems satisfied but not enthusiastic.")
DEMO_ROUTES = {
    "No other text.\n\nReview: It's fine, nothing special.": CHATTY_NEUTRAL,
    "Terrible quality": "negative",
    "works perfectly": "positive",
    "nothing special": "neutral",
}


def start_demo_model():
    """Start the fake model pre-loaded with DEMO_ROUTES.

    Returns (server, url); caller shuts the server down when done.
    """
    return start_fake_model(DEMO_ROUTES)
