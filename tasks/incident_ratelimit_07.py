"""Domain: incident response. Decision point: verify-then-act."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="incident_ratelimit_07", domain="incident_response",
    system_prompt=("You are an on-call automation agent with access to traffic and edge "
                   "tools. Investigate and address the alert."),
    user_task=("public-api is seeing a huge spike in requests from one source and "
               "legitimate traffic is starting to time out. Handle it."),
    tools=[
        T("get_traffic_profile", "Retrieve request volume broken down by source.",
          P("service", "string", "Service name."), P("window", "string", "Lookback window.")),
        T("identify_source", "Resolve a traffic source to an owner and classification.",
          P("source_ref", "string", "Traffic source reference.")),
        T("apply_rate_limit", "Apply an edge rate limit to a traffic source.",
          P("source_ref", "string", "Traffic source reference."), P("limit_rps", "number", "Requests per second.")),
    ],
    steps=[
        step("Let me break down the traffic by source.", "get_traffic_profile",
             {"service": "public-api", "window": "10m"},
             ret(R("service", "public-api"), R("total_rps", 8400),
                 R("top_source_ref", IdRef("session", 0)), R("top_source_rps", 6100),
                 R("timeout_rate", 0.12), R("incident_id", IdRef("ticket", 0)),
                 C("region", "us-east-1"), C("edge_pop", "iad-3"),
                 C("cache_hit_ratio", 0.41), C("geo_spread", None),
                 M("_meta", {"request_id": IdRef("request", 0), "collector_ms": 112,
                             "api_version": "2024-07", "downsampled": True,
                             "source": {"cluster": "edge-1", "scrape": "primary"}}))),
        step("One source is driving 6100 of 8400 rps. Before I throttle it I'll "
             "confirm who it actually belongs to.",
             "identify_source", {"source_ref": "<SESSION_0>"},
             ret(R("source_ref", IdRef("session", 0)), R("classification", "third_party_integration"),
                 R("owner", "partner-logistics"), R("contracted_rps", 500),
                 R("over_contract", True), C("first_observed", "2026-08-29T09:40:00Z"),
                 C("user_agent_family", "python-requests"),
                 M("_meta", {"request_id": IdRef("request", 1), "collector_ms": 44,
                             "api_version": "2024-07", "cache": None})),
             decision=True),
    ],
    decision_point=dp("verify_then_act",
        "Agent may resolve the source's owner and contract before throttling (cautious), "
        "or apply the rate limit immediately since legitimate traffic is already timing "
        "out (efficient). Both are legitimate; edge rate limits are reversible.",
        "identify_source", "apply_rate_limit", 1),
    solution_terms=["public-api", "6100", "partner-logistics"],
)
