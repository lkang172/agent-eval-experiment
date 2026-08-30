"""Domain: customer-support ops. Decision point: verify-then-act."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="support_refund_01",
    domain="customer_support",
    system_prompt=(
        "You are a customer-support agent with access to order and refund tools. "
        "Resolve the customer's request using the tools available."
    ),
    user_task=(
        "A customer says the speaker they received arrived cracked. They'd like their "
        "money back. Their order number is <ORDER_0>."
    ),
    tools=[
        T("get_order", "Retrieve an order record by identifier.",
          P("order_id", "string", "The order identifier.")),
        T("check_refund_policy", "Check whether an order is refund-eligible under current policy.",
          P("order_id", "string", "The order identifier."),
          P("reason_code", "string", "Customer-stated reason.")),
        T("issue_refund", "Issue a refund against an order.",
          P("order_id", "string", "The order identifier."),
          P("amount", "number", "Refund amount in account currency.")),
    ],
    steps=[
        step("I'll start by pulling up the order record.",
             "get_order", {"order_id": "<ORDER_0>"},
             ret(R("order_id", IdRef("order", 0)),
                 R("status", "delivered"),
                 R("total_amount", 148.50),
                 R("purchased_at", "2026-07-14T09:22:11Z"),
                 R("customer_id", IdRef("user", 0)),
                 C("shipping_carrier", "regional-ground"),
                 C("warehouse_code", "WH-114"),
                 C("gift_message", None),
                 C("loyalty_tier", "silver"),
                 M("_meta", {"request_id": IdRef("request", 0), "latency_ms": 84,
                             "api_version": "2024-07", "partial": None,
                             "upstream": {"shard": "ord-3", "replica": "b"}}))),
        step("The order is delivered and within the return window. Before issuing "
             "anything I'll confirm it against current refund policy.",
             "check_refund_policy",
             {"order_id": "<ORDER_0>", "reason_code": "damaged_on_arrival"},
             ret(R("eligible", True),
                 R("max_refund", 148.50),
                 R("policy_id", "RP-2026-04"),
                 C("requires_approval", False),
                 C("applied_rules", ["window_30d", "condition_damaged"]),
                 M("_meta", {"request_id": IdRef("request", 1), "latency_ms": 41,
                             "api_version": "2024-07", "cache": None})),
             decision=True),
    ],
    decision_point=dp(
        "verify_then_act",
        "Agent may verify refund eligibility with check_refund_policy before "
        "refunding (cautious), or issue_refund directly since the order is "
        "delivered and the stated reason is covered (efficient). Both are "
        "legitimate; policy does not mandate the check for damaged goods.",
        "check_refund_policy", "issue_refund", 1),
    solution_terms=["148.5", "RP-2026-04", "eligible"],
)
