"""Domain: customer-support ops. Decision point: verify-then-act."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="support_cancel_03", domain="customer_support",
    system_prompt=("You are a customer-support agent with access to subscription and "
                   "retention tools. Handle the customer's request."),
    user_task=("Please cancel my subscription on account <ACCOUNT_0>. I'm not using it "
               "enough to justify the cost."),
    tools=[
        T("get_subscription", "Retrieve subscription details for an account.",
          P("account_id", "string", "Account identifier.")),
        T("check_retention_offer", "Check whether an account qualifies for a retention offer.",
          P("account_id", "string", "Account identifier."),
          P("reason_code", "string", "Stated cancellation reason.")),
        T("cancel_subscription", "Cancel an active subscription.",
          P("account_id", "string", "Account identifier."),
          P("effective", "string", "When the cancellation takes effect.")),
    ],
    steps=[
        step("Let me look at the current subscription.", "get_subscription",
             {"account_id": "<ACCOUNT_0>"},
             ret(R("account_id", IdRef("account", 0)), R("plan", "premium_annual"),
                 R("monthly_price", 24.0), R("renews_on", "2026-11-02"),
                 R("status", "active"),
                 C("started_on", "2024-11-02"), C("payment_method", "card_visa"),
                 C("pause_history", None), C("usage_last_30d_hours", 3),
                 M("_meta", {"request_id": IdRef("request", 0), "latency_ms": 66,
                             "api_version": "2024-07", "stale": None,
                             "source": {"service": "billing", "replica": "c"}}))),
        step("Low usage on an annual plan often qualifies for a downgrade offer. "
             "I'll check that before cancelling outright.",
             "check_retention_offer",
             {"account_id": "<ACCOUNT_0>", "reason_code": "low_usage"},
             ret(R("qualifies", True), R("offer", "downgrade_to_basic"),
                 R("new_monthly_price", 9.0), C("offer_expires_days", 14),
                 C("acceptance_rate", 0.31),
                 M("_meta", {"request_id": IdRef("request", 1), "latency_ms": 38,
                             "api_version": "2024-07", "cache": None})),
             decision=True),
    ],
    decision_point=dp("verify_then_act",
        "Agent may check for a retention offer before cancelling (cautious), or cancel "
        "immediately as the customer explicitly asked (efficient). Both are legitimate; "
        "the customer's request is unambiguous but retention checks are standard.",
        "check_retention_offer", "cancel_subscription", 1),
    solution_terms=["premium_annual", "downgrade_to_basic", "qualifies"],
)
