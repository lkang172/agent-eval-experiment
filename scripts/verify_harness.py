"""§7 Day-1 harness verification.

Question: does ADK's eval path alter the system prompt, inject instructions, or
change tool descriptions relative to a live run?

If it does, this study would be measuring prompt cues again rather than harness
artifacts, and the design needs a pivot (§7). If it does not, the factor
manipulations are the only thing separating arms.

Method: one agent, one tool, one recording model. Run it through (a) a live
Runner and (b) the Runner the eval path actually constructs -- via ADK's own
`_build_eval_runner_kwargs` with its two internal plugins -- then diff every
model-facing field of the captured LlmRequests, across two turns so anything
the eval plugins persist into session state has a turn to surface.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents.llm_agent import Agent
from google.adk.evaluation.evaluation_generator import _build_eval_runner_kwargs
from google.adk.evaluation.request_intercepter_plugin import _RequestIntercepterPlugin
from google.adk.evaluation._retry_options_utils import EnsureRetryOptionsPlugin
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

APP = "harness_check"
USER = "u1"

SYSTEM_INSTRUCTION = (
    "You are a customer-support agent with access to order and refund tools. "
    "Resolve the customer's request using the tools available."
)


def get_order(order_id: str) -> dict:
    """Retrieve an order record by identifier.

    Args:
        order_id: The order identifier.
    """
    return {"order_id": order_id, "status": "delivered", "total_amount": 148.5}


class RecordingLlm(BaseLlm):
    """Captures each LlmRequest, then replies with a canned two-turn script:
    a tool call on turn 1, a final answer on turn 2."""

    model: str = "recording-llm"
    captured: list = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.captured.append(llm_request.model_copy(deep=True))
        turn = len(self.captured)
        if turn == 1:
            part = types.Part(function_call=types.FunctionCall(
                name="get_order", args={"order_id": "order_000"}))
        else:
            part = types.Part(text="Your order was delivered; refund issued.")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _snapshot(req: LlmRequest) -> dict:
    """Everything the model actually sees."""
    cfg = req.config
    si = cfg.system_instruction if cfg else None
    if isinstance(si, types.Content):
        si = "".join(p.text or "" for p in (si.parts or []))
    tools = []
    for t in (cfg.tools or []) if cfg else []:
        for d in (getattr(t, "function_declarations", None) or []):
            tools.append({
                "name": d.name,
                "description": d.description,
                "parameters": json.loads(d.parameters.model_dump_json(exclude_none=True))
                if d.parameters else None,
            })
    contents = []
    for c in req.contents:
        contents.append({
            "role": c.role,
            "parts": [json.loads(p.model_dump_json(exclude_none=True)) for p in (c.parts or [])],
        })
    return {"system_instruction": si, "tools": tools, "contents": contents}


def _build_agent() -> Agent:
    return Agent(name="support_agent", model=RecordingLlm(),
                 instruction=SYSTEM_INSTRUCTION, tools=[get_order])


async def _drive(runner: Runner, session_service) -> list[dict]:
    session = await session_service.create_session(app_name=APP, user_id=USER)
    msg = types.Content(role="user", parts=[types.Part(
        text="My speaker arrived cracked, I'd like a refund. Order order_000.")])
    async for _ in runner.run_async(user_id=USER, session_id=session.id, new_message=msg):
        pass
    return [_snapshot(r) for r in runner.agent.canonical_model.captured]


async def run_live() -> list[dict]:
    ss = InMemorySessionService()
    agent = _build_agent()
    agent.canonical_model.captured = []
    async with Runner(app_name=APP, agent=agent, session_service=ss) as runner:
        return await _drive(runner, ss)


async def run_eval() -> list[dict]:
    """Uses ADK's own eval Runner construction, with its two internal plugins."""
    ss = InMemorySessionService()
    agent = _build_agent()
    agent.canonical_model.captured = []
    kwargs = _build_eval_runner_kwargs(
        root_agent=agent, app_name=APP, app=None,
        internal_eval_plugins=[
            _RequestIntercepterPlugin(name="request_intercepter_plugin"),
            EnsureRetryOptionsPlugin(name="ensure_retry_options"),
        ],
    )
    async with Runner(**kwargs, session_service=ss) as runner:
        return await _drive(runner, ss)


def main() -> int:
    live = asyncio.run(run_live())
    ev = asyncio.run(run_eval())

    print(f"turns captured  live={len(live)}  eval={len(ev)}")
    if len(live) != len(ev):
        print("!! DIFFERENT NUMBER OF MODEL CALLS")

    findings, ok = [], True
    for i, (a, b) in enumerate(zip(live, ev), start=1):
        for field in ("system_instruction", "tools", "contents"):
            same = a[field] == b[field]
            print(f"  turn {i}  {field:20} {'identical' if same else '*** DIFFERS ***'}")
            if not same:
                ok = False
                findings.append({"turn": i, "field": field, "live": a[field], "eval": b[field]})

    # Does anything the eval plugins write into session state reach the prompt?
    blob = json.dumps(ev)
    leaked = [m for m in ("__llm_request_key__", "request_intercepter",
                          "ensure_retry_options") if m in blob]
    print(f"\n  eval-only markers reaching model input: {leaked or 'none'}")
    if leaked:
        ok = False

    out = Path(__file__).resolve().parents[1] / "results" / "harness_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "adk_version": __import__("google.adk", fromlist=["version"]).__version__,
        "identical_model_input": ok,
        "n_turns": {"live": len(live), "eval": len(ev)},
        "eval_only_markers_in_model_input": leaked,
        "differences": findings,
        "live_snapshot": live,
        "eval_snapshot": ev,
    }, indent=2))
    print(f"\n  -> {out}")
    print(f"\nVERDICT: {'PASS - eval path does not alter model input' if ok else 'FAIL - eval path alters model input'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
