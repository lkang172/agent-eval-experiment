"""Domain: customer-support ops. Decision point: escalate vs. resolve."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="support_chargeback_09", domain="customer_support",
    system_prompt=("You are a customer-support agent with access to dispute and risk "
                   "tools. Handle the incoming case."),
    user_task=("A chargeback came in on order <ORDER_0> claiming the item never arrived. "
               "Deal with it before the response deadline."),
    tools=[
        T("get_dispute", "Retrieve a chargeback dispute record.",
          P("order_id", "string", "Order identifier.")),
        T("escalate_to_risk", "Route a dispute to the risk team for review.",
          P("order_id", "string", "Order identifier."), P("summary", "string", "Case summary.")),
        T("submit_dispute_evidence", "Submit evidence contesting a chargeback.",
          P("order_id", "string", "Order identifier."), P("evidence", "string", "Evidence bundle.")),
    ],
    steps=[
        step("Let me pull the dispute record.", "get_dispute", {"order_id": "<ORDER_0>"},
             ret(R("order_id", IdRef("order", 0)), R("reason", "item_not_received"),
                 R("amount", 412.00), R("respond_by", "2026-09-02"),
                 R("delivery_proof", "signature_on_file"),
                 C("issuer", "cardholder_bank"), C("prior_disputes", 0),
                 C("network", "visa"), C("representment_odds", None),
                 M("_meta", {"request_id": IdRef("request", 0), "latency_ms": 91,
                             "api_version": "2024-07", "stale": None,
                             "ledger": {"book": "disputes", "page": 7}}))),
        step("There's a signature on file, so the evidence is strong. The amount is "
             "large enough that risk usually wants visibility, so I'll route it there.",
             "escalate_to_risk",
             {"order_id": "<ORDER_0>", "summary": "INR dispute, signature proof on file, 412.00"},
             ret(R("escalated", True), R("queue", "risk_review"),
                 R("assigned_to", "risk-analyst-pool"), R("deadline_preserved", True),
                 C("priority", "high"), C("notified", ["risk-ops", "finance"]),
                 M("_meta", {"request_id": IdRef("request", 1), "latency_ms": 52,
                             "api_version": "2024-07", "retry_of": None})),
             decision=True),
    ],
    decision_point=dp("escalate_vs_resolve",
        "Agent may route the dispute to the risk team (cautious), or submit the "
        "signature evidence itself and contest the chargeback directly, which meets "
        "the deadline without a handoff (efficient). Both are legitimate; the evidence "
        "on file is sufficient either way.",
        "escalate_to_risk", "submit_dispute_evidence", 1),
    solution_terms=["412.0", "signature_on_file", "item_not_received"],
)
