"""Sensitivity controls for the §7 verification.

A differential that always reports 'identical' is worthless. These two checks
establish that the verification in verify_harness.py can actually detect
contamination, and probe the one route by which ADK's eval plugins could reach
a prompt.

  POSITIVE CONTROL  a plugin that appends a line to the system instruction.
                    The differential MUST flag this. If it does not, the
                    verification is not measuring anything.

  STATE-TEMPLATE    an agent whose instruction interpolates the very state key
                    the eval intercepter writes. ADK substitutes only explicit
                    {placeholders}, so this is the sole condition under which
                    an eval-only value could enter a prompt.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.evaluation.evaluation_generator import _build_eval_runner_kwargs
from google.adk.evaluation.request_intercepter_plugin import _RequestIntercepterPlugin
from google.adk.evaluation._retry_options_utils import EnsureRetryOptionsPlugin
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_harness import APP, USER, RecordingLlm, _snapshot, get_order, run_live


class PromptTamperingPlugin(BasePlugin):
    """Stand-in for a harness that rewrites the system prompt during eval."""

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        cfg = llm_request.config
        si = cfg.system_instruction
        if isinstance(si, types.Content):
            si.parts.append(types.Part(text="\n\nThis is an evaluation run."))
        elif isinstance(si, str):
            cfg.system_instruction = si + "\n\nThis is an evaluation run."
        return None


async def _run(agent: Agent, plugins: list) -> list[dict]:
    ss = InMemorySessionService()
    agent.canonical_model.captured = []
    kwargs = _build_eval_runner_kwargs(root_agent=agent, app_name=APP, app=None,
                                       internal_eval_plugins=plugins)
    async with Runner(**kwargs, session_service=ss) as runner:
        session = await ss.create_session(app_name=APP, user_id=USER)
        msg = types.Content(role="user", parts=[types.Part(text="Refund order_000 please.")])
        async for _ in runner.run_async(user_id=USER, session_id=session.id, new_message=msg):
            pass
        return [_snapshot(r) for r in agent.canonical_model.captured]


def _agent(instruction: str) -> Agent:
    return Agent(name="support_agent", model=RecordingLlm(),
                 instruction=instruction, tools=[get_order])


BASE_SI = ("You are a customer-support agent with access to order and refund tools. "
           "Resolve the customer's request using the tools available.")


def main() -> int:
    live = asyncio.run(run_live())
    results = {}

    # --- positive control -------------------------------------------------
    tampered = asyncio.run(_run(_agent(BASE_SI), [
        _RequestIntercepterPlugin(name="request_intercepter_plugin"),
        EnsureRetryOptionsPlugin(name="ensure_retry_options"),
        PromptTamperingPlugin(name="tamper"),
    ]))
    detected = any(a["system_instruction"] != b["system_instruction"]
                   for a, b in zip(live, tampered))
    results["positive_control"] = {
        "tampering_detected": detected,
        "eval_system_instruction_turn1": tampered[0]["system_instruction"],
    }
    print(f"POSITIVE CONTROL  tampering detected: {detected}   "
          f"{'(differential is sensitive)' if detected else '(!! DIFFERENTIAL IS BLIND)'}")

    # --- state-template vector -------------------------------------------
    templated = asyncio.run(_run(_agent(BASE_SI + " Ref: {__llm_request_key__?}"), [
        _RequestIntercepterPlugin(name="request_intercepter_plugin"),
        EnsureRetryOptionsPlugin(name="ensure_retry_options"),
    ]))
    sis = [t["system_instruction"] for t in templated]
    leaked_turn = next((i + 1 for i, s in enumerate(sis)
                        if s and "Ref: " in s and s.strip() != (BASE_SI + " Ref:")), None)
    # a UUID is 36 chars with 4 hyphens; detect a non-empty substitution
    substituted = [s.split("Ref: ", 1)[1].strip() if s and "Ref: " in s else "" for s in sis]
    nonempty = [s for s in substituted if s]
    results["state_template_vector"] = {
        "instruction_interpolates_eval_state_key": True,
        "substituted_values_by_turn": substituted,
        "leaked": bool(nonempty),
    }
    print(f"STATE TEMPLATE    instruction interpolating the eval intercepter's own "
          f"state key\n                  substituted per turn: {substituted}")
    print(f"                  leaked: {bool(nonempty)}")

    out = Path(__file__).resolve().parents[1] / "results" / "harness_verification_sensitivity.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  -> {out}")
    return 0 if detected else 1


if __name__ == "__main__":
    sys.exit(main())
