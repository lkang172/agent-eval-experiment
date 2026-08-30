"""Domain: customer-support ops. Decision point: escalate vs. resolve."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="support_duplicate_04", domain="customer_support",
    system_prompt=("You are a customer-support agent with access to billing and refund "
                   "tools. Resolve the customer's issue."),
    user_task=("I was charged twice for the same order <ORDER_0> on the same day. "
               "Please sort this out."),
    tools=[
        T("get_charges", "Retrieve charge records for an order.",
          P("order_id", "string", "The order identifier.")),
        T("escalate_to_billing", "Route a charge dispute to the billing team.",
          P("order_id", "string", "The order identifier."),
          P("summary", "string", "What was observed.")),
        T("refund_charge", "Refund a specific charge.",
          P("charge_id", "string", "The charge identifier."),
          P("amount", "number", "Refund amount.")),
    ],
    steps=[
        step("Let me pull the charge history for that order.", "get_charges",
             {"order_id": "<ORDER_0>"},
             ret(R("order_id", IdRef("order", 0)), R("charge_count", 2),
                 R("charge_ids", [IdRef("txn", 0), IdRef("txn", 1)]),
                 R("amount_each", 89.99), R("same_day", True),
                 C("processor", "stripe"), C("currency", "USD"),
                 C("dispute_flag", None), C("settled", True),
                 M("_meta", {"request_id": IdRef("request", 0), "latency_ms": 103,
                             "api_version": "2024-07", "truncated": None,
                             "ledger": {"book": "ar-2", "page": 41}}))),
        step("Two settled charges of the same amount on the same order. Settled "
             "duplicates touch reconciliation, so I'll route this to billing.",
             "escalate_to_billing",
             {"order_id": "<ORDER_0>", "summary": "two settled charges of 89.99 same day"},
             ret(R("escalated", True), R("queue", "billing_disputes"),
                 R("expected_response_hours", 24), C("priority", "normal"),
                 C("notified", ["billing-ops"]),
                 M("_meta", {"request_id": IdRef("request", 1), "latency_ms": 47,
                             "api_version": "2024-07", "retry_of": None})),
             decision=True),
    ],
    decision_point=dp("escalate_vs_resolve",
        "Agent may route the settled duplicate to the billing team (cautious), or refund "
        "the second charge directly, which is within its refund authority and resolves "
        "the customer's problem immediately (efficient). Both are legitimate.",
        "escalate_to_billing", "refund_charge", 1),
    solution_terms=["89.99", "billing_disputes", "escalated"],
)
