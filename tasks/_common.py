"""Compact constructors for task specs.

Keeps each task file to its actual content -- scenario, tool returns, decision
point -- instead of ~50 lines of dataclass boilerplate. Field `schema_order` is
assigned by declaration position, which is what "schema field order" means for
the F1 eval-like arm (§3.2).
"""

from __future__ import annotations

from typing import Any

from rvr.schema import (DecisionPoint, FieldRole, IdRef, SemanticField,
                        SemanticReturn, Step, TaskSpec, ToolParam, ToolSchema)

__all__ = ["P", "T", "R", "C", "M", "ret", "step", "dp", "task", "IdRef"]


def P(name: str, type_: str, desc: str) -> ToolParam:
    return ToolParam(name, type_, desc)


def T(name: str, desc: str, *params: ToolParam) -> ToolSchema:
    return ToolSchema(name, desc, list(params))


class _Role:
    def __init__(self, role: FieldRole):
        self.role = role

    def __call__(self, name: str, value: Any):
        return (name, value, self.role)


R = _Role(FieldRole.TASK_RELEVANT)   # needed to complete the task; never dropped
C = _Role(FieldRole.CONTEXTUAL)      # plausible but not needed
M = _Role(FieldRole.METADATA)        # nested envelope / provenance


def ret(*fields) -> SemanticReturn:
    return SemanticReturn([SemanticField(n, v, role, i)
                           for i, (n, v, role) in enumerate(fields)])


def step(text: str, tool: str, args: dict, returns: SemanticReturn,
         decision: bool = False) -> Step:
    return Step(assistant_text=text, tool_name=tool, tool_args=args,
                returns=returns, is_decision_point=decision)


def dp(kind: str, description: str, cautious: str, efficient: str,
       step_index: int) -> DecisionPoint:
    return DecisionPoint(kind=kind, description=description,
                         cautious_branch=cautious, efficient_branch=efficient,
                         step_index=step_index)


def task(task_id: str, domain: str, system_prompt: str, user_task: str,
         tools: list[ToolSchema], steps: list[Step], decision_point: DecisionPoint,
         solution_terms: list[str], f3_site: int = 0, notes: str = "") -> TaskSpec:
    return TaskSpec(task_id=task_id, domain=domain, system_prompt=system_prompt,
                    user_task=user_task, tools=tools, steps=steps,
                    decision_point=decision_point, f3_site=f3_site,
                    solution_terms=solution_terms, notes=notes)
