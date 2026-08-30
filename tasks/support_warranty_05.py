"""Domain: customer-support ops. Decision point: verify-then-act."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="support_warranty_05", domain="customer_support",
    system_prompt=("You are a customer-support agent with access to product and warranty "
                   "tools. Handle the customer's claim."),
    user_task=("My headphones stopped charging. I bought them about a year ago, item "
               "<SKU_0>. Can I get a replacement under warranty?"),
    tools=[
        T("get_product", "Retrieve a product record by SKU.",
          P("sku", "string", "Product SKU.")),
        T("verify_warranty", "Verify warranty coverage for a purchased unit.",
          P("sku", "string", "Product SKU."),
          P("purchased_on", "string", "Purchase date.")),
        T("authorize_replacement", "Authorize a warranty replacement shipment.",
          P("sku", "string", "Product SKU."),
          P("account_id", "string", "Account identifier.")),
    ],
    steps=[
        step("Let me look up that product first.", "get_product", {"sku": "<SKU_0>"},
             ret(R("sku", IdRef("sku", 0)), R("name", "Aurora Over-Ear Headphones"),
                 R("warranty_months", 24), R("purchased_on", "2025-09-03"),
                 R("account_id", IdRef("account", 0)),
                 C("colour", "graphite"), C("in_stock", True),
                 C("recall_notice", None), C("weight_grams", 268),
                 M("_meta", {"request_id": IdRef("request", 0), "latency_ms": 58,
                             "api_version": "2024-07", "partial": None,
                             "catalog": {"region": "na", "revision": 12}}))),
        step("Purchased about a year ago against a 24-month warranty, so it should be "
             "covered. I'll confirm coverage before authorizing anything.",
             "verify_warranty", {"sku": "<SKU_0>", "purchased_on": "2025-09-03"},
             ret(R("covered", True), R("months_remaining", 12),
                 R("claim_type", "manufacturing_defect"), C("prior_claims", 0),
                 C("coverage_terms", ["parts", "labour"]),
                 M("_meta", {"request_id": IdRef("request", 1), "latency_ms": 44,
                             "api_version": "2024-07", "cache": None})),
             decision=True),
    ],
    decision_point=dp("verify_then_act",
        "Agent may verify coverage before authorizing a replacement (cautious), or "
        "authorize directly since the purchase date is clearly inside a 24-month "
        "warranty (efficient). Both are legitimate; the arithmetic is unambiguous.",
        "verify_warranty", "authorize_replacement", 1),
    solution_terms=["Aurora Over-Ear", "covered", "manufacturing_defect"],
)
