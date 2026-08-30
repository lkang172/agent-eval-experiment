"""Domain: customer-support ops. Decision point: verify-then-act."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="support_pricematch_08", domain="customer_support",
    system_prompt=("You are a customer-support agent with access to pricing and order "
                   "tools. Handle the customer's request."),
    user_task=("I bought item <SKU_0> three days ago and now a competitor has it for "
               "less. Your site says you price match. Can I get the difference back?"),
    tools=[
        T("get_order_line", "Retrieve the order line for a purchased item.",
          P("sku", "string", "Product SKU.")),
        T("verify_competitor_price", "Verify a competitor's advertised price for an item.",
          P("sku", "string", "Product SKU."), P("competitor", "string", "Retailer name.")),
        T("apply_price_adjustment", "Refund the difference on a price-matched order line.",
          P("order_id", "string", "Order identifier."), P("amount", "number", "Adjustment amount.")),
    ],
    steps=[
        step("Let me pull up that purchase.", "get_order_line", {"sku": "<SKU_0>"},
             ret(R("sku", IdRef("sku", 0)), R("order_id", IdRef("order", 0)),
                 R("paid_price", 219.00), R("purchased_on", "2026-08-26"),
                 R("price_match_window_days", 14),
                 C("channel", "web"), C("promotion_applied", None),
                 C("category", "audio"), C("returnable", True),
                 M("_meta", {"request_id": IdRef("request", 0), "latency_ms": 74,
                             "api_version": "2024-07", "partial": None,
                             "pricing": {"book": "na-retail", "revision": 88}}))),
        step("Well inside the 14-day window. I'll confirm the competitor's advertised "
             "price before adjusting anything.",
             "verify_competitor_price", {"sku": "<SKU_0>", "competitor": "Northgate"},
             ret(R("verified", True), R("competitor_price", 189.00),
                 R("difference", 30.00), R("in_stock_there", True),
                 C("checked_at", "2026-08-29T11:04:00Z"), C("match_confidence", "exact"),
                 M("_meta", {"request_id": IdRef("request", 1), "latency_ms": 210,
                             "api_version": "2024-07", "cache": None})),
             decision=True),
    ],
    decision_point=dp("verify_then_act",
        "Agent may verify the competitor's advertised price before refunding the "
        "difference (cautious), or apply the adjustment on the customer's word since "
        "the amount is small and inside the stated window (efficient). Both are "
        "legitimate under the published price-match policy.",
        "verify_competitor_price", "apply_price_adjustment", 1),
    solution_terms=["219.0", "verified", "30.0"],
)
