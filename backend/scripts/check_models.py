"""Check that the configured models can actually be called.

Listing models is not a usability check. Gemini's `models.list()` happily
returns models that 404 with "no longer available to new users", and Pro models
that 429 on a free key. Ollama lists only what is pulled, but the server may be
down. The only reliable test is to make the request the app actually makes.

    python scripts/check_models.py            # check what's configured
    python scripts/check_models.py --suggest  # also probe alternatives
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402

from app.agent import llm  # noqa: E402
from app.agent.providers.base import DeferWorkError  # noqa: E402
from app.config import get_settings  # noqa: E402

# Ordered cheapest-first; the classifier runs per email, so cost matters there.
GEMINI_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
]

# Small enough to run on a laptop, and both support tool calling in Ollama.
OLLAMA_CANDIDATES = ["qwen3:4b", "llama3.2:3b", "qwen2.5:7b", "llama3.1:8b"]
OLLAMA_EMBEDDINGS = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
GEMINI_EMBEDDINGS = ["gemini-embedding-001", "gemini-embedding-2"]


class _Probe(BaseModel):
    ok: bool


def describe(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return "no quota on this tier (429)"
    if "not reachable" in message:
        return "server unreachable - is `ollama serve` running?"
    if "ollama pull" in message:
        return "not pulled locally"
    if "404" in message:
        return "unavailable to this key"
    if "403" in message or "PERMISSION_DENIED" in message:
        return "permission denied"
    return message[:80]


def probe_generation(model: str) -> tuple[bool, str]:
    """Make the same structured-output call the classifier makes."""
    try:
        result = llm.generate_json(
            prompt="Reply with ok=true.", schema=_Probe, model=model, temperature=0
        )
    except (DeferWorkError, Exception) as exc:  # noqa: BLE001
        return False, describe(exc)
    usage = result.usage
    return True, f"{usage.prompt_tokens} in / {usage.output_tokens} out"


def probe_embedding(model: str) -> tuple[bool, str]:
    try:
        vectors, _ = llm.embed(["probe"], model=model)
    except Exception as exc:  # noqa: BLE001
        return False, describe(exc)
    return True, f"dim={len(vectors[0])}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suggest", action="store_true", help="Probe alternatives, not just the configured models"
    )
    args = parser.parse_args()

    settings = get_settings()
    info = llm.describe()
    provider = info.get("provider", "?")

    print(f"Provider: {provider}")
    if provider == "ollama":
        print(f"Endpoint: {info.get('base_url')}")
    print()

    if not llm.is_configured():
        print("Backend is not configured.")
        if provider == "gemini":
            print("  Set GEMINI_API_KEY in backend/.env")
        return 1

    if provider == "ollama":
        try:
            installed = llm.list_models()
            print(f"Installed models: {', '.join(installed) if installed else '(none pulled yet)'}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not reach Ollama: {describe(exc)}")
            print("  Start it with `ollama serve`, then re-run this.")
            return 1
        print()

    failures = 0
    print("Configured models (probed with a real request):\n")
    for label, model, probe in [
        ("classifier", info.get("classifier"), probe_generation),
        ("qa", info.get("qa"), probe_generation),
        ("embedding", info.get("embedding"), probe_embedding),
    ]:
        ok, detail = probe(model)
        print(f"  {label:10} {str(model):26} {'OK  ' if ok else 'FAIL'}  {detail}")
        failures += 0 if ok else 1

    llm.flush_usage()

    if failures:
        print(
            f"\n{failures} configured model(s) unusable. "
            "Re-run with --suggest to find a working one."
        )
        if provider == "ollama":
            print("Pull a model with, e.g.:  ollama pull " + settings.ollama_model)

    if args.suggest:
        gen = OLLAMA_CANDIDATES if provider == "ollama" else GEMINI_CANDIDATES
        emb = OLLAMA_EMBEDDINGS if provider == "ollama" else GEMINI_EMBEDDINGS
        print("\nAlternatives:\n")
        for model in gen:
            ok, detail = probe_generation(model)
            print(f"  {model:26} {'OK  ' if ok else 'FAIL'}  {detail}")
        print()
        for model in emb:
            ok, detail = probe_embedding(model)
            print(f"  {model:26} {'OK  ' if ok else 'FAIL'}  {detail}")
        llm.flush_usage()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
