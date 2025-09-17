from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import requests


def run_chat(host: str, model: str, prompt: str, timeout: int = 30) -> Dict[str, Any]:
    """Send a single-turn chat request to Ollama."""
    url = host.rstrip("/") + "/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a test harness checking connectivity."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Base URL for the Ollama server (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Model name to query (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default="Hello from the connectivity test!",
        help="Prompt to send with the request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: %(default)s)",
    )

    args = parser.parse_args(argv)

    try:
        data = run_chat(args.host, args.model, args.prompt, timeout=args.timeout)
    except requests.exceptions.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    status = data.get("done", False)
    message = data.get("message", {}).get("content")
    print("--- Ollama connectivity test ---")
    print(f"Host    : {args.host}")
    print(f"Model   : {args.model}")
    print(f"Prompt  : {args.prompt}")
    print(f"Success : {status}")
    if message:
        print("\nResponse snippet:\n")
        print(message.strip())
    else:
        print("\nNo content returned. Full payload follows:\n")
        print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

