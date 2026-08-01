"""Check that the configured Gemini models can actually be called.

`client.models.list()` is not a usability check — it happily lists models that
return `404 ... no longer available to new users` the moment you call them, and
Pro models that return `429` on a free-tier key. The only reliable test is to
make the request the app actually makes.

    python scripts/check_models.py            # check what's in .env
    python scripts/check_models.py --suggest  # also probe alternatives
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402

from app.agent import llm  # noqa: E402
from app.config import get_settings  # noqa: E402

# Ordered cheapest-first; the classifier runs per email, so cost matters there.
GENERATION_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
]

EMBEDDING_CANDIDATES = ["gemini-embedding-001", "gemini-embedding-2"]


class _Probe(BaseModel):
    ok: bool


def describe(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    if "404" in message:
        return "unavailable to this key"
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return "no quota on this tier (429)"
    if "403" in message or "PERMISSION_DENIED" in message:
        return "permission denied"
    return message[:80]


def probe_generation(model: str) -> tuple[bool, str]:
    """Make the same structured-output call the classifier makes."""
    try:
        result = llm.generate_json(
            prompt="Reply with ok=true.", schema=_Probe, model=model, temperature=0
        )
    except Exception as exc:  # noqa: BLE001
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

    if not llm.is_configured():
        print("GEMINI_API_KEY is not set in backend/.env")
        return 1

    settings = get_settings()
    failures = 0

    print("Configured models (probed with a real request):\n")
    for label, model, probe in [
        ("classifier", settings.classifier_model, probe_generation),
        ("qa", settings.qa_model, probe_generation),
        ("embedding", settings.embedding_model, probe_embedding),
    ]:
        ok, detail = probe(model)
        print(f"  {label:10} {model:26} {'OK  ' if ok else 'FAIL'}  {detail}")
        failures += 0 if ok else 1

    if failures:
        print(
            f"\n{failures} configured model(s) unusable. "
            "Re-run with --suggest to find a working one."
        )

    if args.suggest:
        print("\nAlternatives:\n")
        for model in GENERATION_CANDIDATES:
            ok, detail = probe_generation(model)
            print(f"  {model:26} {'OK  ' if ok else 'FAIL'}  {detail}")
        print()
        for model in EMBEDDING_CANDIDATES:
            ok, detail = probe_embedding(model)
            print(f"  {model:26} {'OK  ' if ok else 'FAIL'}  {detail}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
