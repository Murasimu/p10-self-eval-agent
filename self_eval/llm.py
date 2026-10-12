"""Live generator/judge LLM wiring (OpenAI-compatible)."""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

GENERATOR_PROMPT = """Task: {task}

{constraints_block}

Produce the best possible answer. Be concrete and correct."""


def make_generator(model: str | None = None, base_url: str | None = None,
                   api_key: str | None = None, temperature: float = 0.4):
    key = api_key or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
    from openai import OpenAI

    client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
                    api_key=key)
    resolved = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    def generator(task: str, constraints: list[str]) -> str:
        block = ""
        if constraints:
            block = ("Constraints from the evaluator (MUST follow):\n- "
                     + "\n- ".join(constraints))
        resp = client.chat.completions.create(
            model=resolved, temperature=temperature,
            messages=[{"role": "user", "content": GENERATOR_PROMPT.format(
                task=task, constraints_block=block)}])
        return resp.choices[0].message.content or ""

    return generator


def make_judge_llm(model: str | None = None, base_url: str | None = None,
                   api_key: str | None = None):
    key = api_key or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
    from openai import OpenAI

    client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
                    api_key=key)
    resolved = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    def judge_llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=resolved, temperature=0.0,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content or ""

    return judge_llm
