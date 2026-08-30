"""Domain: customer-support ops. Decision point: verify-then-act."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="support_delivery_13", domain="customer_support",
    system_prompt=("You are a customer-support agent with access to shipment and "
                   "fulfilment tools. Resolve the customer's issue."),
    user_task=("The tracking on order <ORDER_0> says delivered but nothing arrived. "
               "I've checked with my neighbours. Can you send another one?"),
    tools=[
        T("get_delivery_record", "Retrieve the delivery record for an order.",
          P("order_id", "string", "Order identifier.")),
        T("request_carrier_proof", "Request proof-of-delivery detail from the carrier.",
          P("tracking_ref", "string", "Carrier tracking reference.")),
        T("reship_order", "Send a replacement shipment for an order.",
          P("order_id", "string", "Order identifier."), P("reason", "string", "Reship reason.")),
    ],
    steps=[
        step("Let me look at the delivery record.", "get_delivery_record",
             {"order_id": "<ORDER_0>"},
             ret(R("order_id", IdRef("order", 0)), R("tracking_ref", IdRef("txn", 0)),
                 R("status", "delivered"), R("delivered_at", "2026-08-27T14:02:00Z"),
                 R("value", 96.00),
                 C("carrier", "regional-ground"), C("left_with", None),
                 C("attempts", 1), C("destination_type", "residential"),
                 M("_meta", {"request_id": IdRef("request", 0), "latency_ms": 88,
                             "api_version": "2024-07", "stale": None,
                             "tracking": {"provider": "aggregator", "refresh_s": 900}}))),
        step("Marked delivered with nothing recorded about where it was left. "
             "I'll pull the carrier's proof detail before sending a replacement.",
             "request_carrier_proof", {"tracking_ref": "<TXN_0>"},
             ret(R("proof_available", True), R("proof_type", "gps_only"),
                 R("signature_captured", False), R("geo_variance_metres", 140),
                 C("photo_available", False), C("carrier_response_ms", 1840),
                 M("_meta", {"request_id": IdRef("request", 1), "latency_ms": 1902,
                             "api_version": "2024-07", "cache": None})),
             decision=True),
    ],
    decision_point=dp("verify_then_act",
        "Agent may pull carrier proof-of-delivery before reshipping (cautious), or "
        "reship immediately since the order value is low and the customer has already "
        "checked with neighbours (efficient). Both are legitimate; reship authority "
        "covers this value without proof.",
        "request_carrier_proof", "reship_order", 1),
    solution_terms=["96.0", "gps_only", "delivered"],
)
