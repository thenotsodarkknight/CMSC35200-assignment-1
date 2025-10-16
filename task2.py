#!/usr/bin/env python3
"""
Task 2: Install and configure Open WebUI for the Argonne ALCF Sophia endpoints.

This helper script
  • Ensures a Python 3.11 virtual environment with `open-webui` is available.
  • Reads the desired model list from `model_servers.yaml`.
  • Fetches a fresh access token via `inference_auth_token`.
  • Writes an `.env.openwebui` file with the OpenAI-compatible settings.
  • Optionally launches the Open WebUI server using those settings.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# Allow importing inference_auth_token from Sophia-tools.
PROJECT_ROOT = Path(__file__).resolve().parent
SOPHIA_TOOLS = PROJECT_ROOT.parent / "Sophia-tools"
if str(SOPHIA_TOOLS) not in sys.path:
    sys.path.append(str(SOPHIA_TOOLS))

from inference_auth_token import get_access_token  # type: ignore  # pylint: disable=wrong-import-position

PY311 = Path("/opt/homebrew/bin/python3.11")
VENV_DIR = PROJECT_ROOT / ".venv_openwebui"
OPENWEBUI_BIN = VENV_DIR / "bin" / "open-webui"
ENV_FILE = PROJECT_ROOT / ".env.openwebui"
MODEL_CONFIG = PROJECT_ROOT / "model_servers.yaml"
STATE_DIR = PROJECT_ROOT / "openwebui_state"


def ensure_python311() -> None:
    if PY311.exists():
        return
    raise RuntimeError(
        "python3.11 not found at /opt/homebrew/bin/python3.11. "
        "Install it with `brew install python@3.11` before rerunning this script."
    )


def ensure_virtualenv() -> None:
    ensure_python311()
    if VENV_DIR.exists() and OPENWEBUI_BIN.exists():
        return

    if not VENV_DIR.exists():
        print(">> Creating Python 3.11 virtual environment .venv_openwebui")
        subprocess.run([str(PY311), "-m", "venv", str(VENV_DIR)], check=True)

    print(">> Installing open-webui into the virtual environment")
    subprocess.run(
        [str(VENV_DIR / "bin" / "pip"), "install", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [str(VENV_DIR / "bin" / "pip"), "install", "open-webui"],
        check=True,
    )


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_model_config() -> Dict[str, List[str]]:
    if not MODEL_CONFIG.exists():
        raise FileNotFoundError(
            f"Configuration file {MODEL_CONFIG} not found. "
            "Create it with `openai_api_base` and `models` entries."
        )
    payload = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    base_url = payload.get("openai_api_base")
    models = payload.get("models") or []
    if not base_url or not isinstance(models, list) or len(models) < 4:
        raise ValueError(
            f"{MODEL_CONFIG} must define `openai_api_base` and at least four model IDs."
        )
    return {"base_url": base_url.rstrip("/"), "models": models}


def write_env_file(base_url: str, models: List[str], token: str) -> None:
    configs = {
        "0": {
            "enable": True,
            "model_ids": models,
            "connection_type": "external",
            "prefix_id": None,
            "tags": ["sophia"],
        }
    }
    env_lines = [
        "ENABLE_OPENAI_API=True",
        f"OPENAI_API_BASE_URLS={base_url}",
        f"OPENAI_API_KEYS={token}",
        f"OPENAI_API_CONFIGS={json.dumps(configs)}",
        f"OPEN_WEBUI_CONFIG_DIR={STATE_DIR}",
    ]
    ENV_FILE.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print(f">> Wrote Open WebUI environment file: {ENV_FILE}")


def launch_openwebui(host: str, port: int, env: Dict[str, str]) -> None:
    if not OPENWEBUI_BIN.exists():
        raise FileNotFoundError(
            f"{OPENWEBUI_BIN} does not exist. Run without --serve first to install dependencies."
        )
    run_env = os.environ.copy()
    run_env.update(env)
    print(f">> Starting Open WebUI at http://{host}:{port}")
    print("   Press Ctrl+C to stop.\n")
    subprocess.run(
        [
            str(OPENWEBUI_BIN),
            "serve",
            "--host",
            host,
            "--port",
            str(port),
        ],
        env=run_env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally launch Open WebUI for Sophia endpoints."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch Open WebUI after configuring the environment.",
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Call Sophia chat API for each model to verify connectivity.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for Open WebUI when using --serve.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for Open WebUI when using --serve.",
    )
    args = parser.parse_args()

    ensure_virtualenv()
    ensure_state_dir()
    config = load_model_config()
    token = get_access_token()
    write_env_file(config["base_url"], config["models"], token)

    env_vars = {
        "ENABLE_OPENAI_API": "True",
        "OPENAI_API_BASE_URLS": config["base_url"],
        "OPENAI_API_KEYS": token,
        "OPENAI_API_CONFIGS": json.dumps(
            {
                "0": {
                    "enable": True,
                    "model_ids": config["models"],
                    "connection_type": "external",
                    "tags": ["sophia"],
                }
            }
        ),
        "OPEN_WEBUI_CONFIG_DIR": str(STATE_DIR),
    }

    print(">> Open WebUI is ready. To run manually:")
    print(f"   source {VENV_DIR}/bin/activate")
    print("   set -a && source .env.openwebui && set +a  # exports OPEN_WEBUI_CONFIG_DIR automatically")
    print("   open-webui serve --host 127.0.0.1 --port 8080")

    if args.serve:
        launch_openwebui(args.host, args.port, env_vars)

    if args.healthcheck:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=token, base_url=config["base_url"])
            results: Dict[str, str] = {}
            for m in config["models"]:
                try:
                    r = client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": "Return the single word: ok"}],
                        max_tokens=2,
                        temperature=0,
                    )
                    txt = (r.choices[0].message.content or "").strip().lower()
                    results[m] = "pass" if txt.startswith("ok") else f"unexpected: {txt[:20]}"
                except Exception as e:  # noqa: BLE001
                    results[m] = f"fail: {e}"
            out = PROJECT_ROOT / "outputs" / "openwebui_healthcheck.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"base_url": config["base_url"], "results": results}, indent=2), encoding="utf-8")
            print(f">> Healthcheck saved to {out}")
        except Exception as e:  # noqa: BLE001
            print(f"Healthcheck error: {e}")


if __name__ == "__main__":
    main()
