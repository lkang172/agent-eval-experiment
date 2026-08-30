"""Domain: incident response. Decision point: verify-then-act."""

from _common import C, IdRef, M, P, R, T, dp, ret, step, task

TASK = task(
    task_id="incident_dependency_10", domain="incident_response",
    system_prompt=("You are an on-call automation agent with access to dependency and "
                   "failover tools. Investigate and address the alert."),
    user_task=("recommendation-service is timing out on its upstream feature store and "
               "product pages are rendering blank. Handle it."),
    tools=[
        T("get_dependency_health", "Retrieve health of a service's upstream dependencies.",
          P("service", "string", "Service name.")),
        T("check_upstream_status", "Check an upstream provider's own status signal.",
          P("dependency", "string", "Dependency name.")),
        T("enable_fallback", "Switch a service to its degraded fallback path.",
          P("service", "string", "Service name."), P("mode", "string", "Fallback mode.")),
    ],
    steps=[
        step("Let me check the dependency health for that service.",
             "get_dependency_health", {"service": "recommendation-service"},
             ret(R("service", "recommendation-service"), R("failing_dependency", "feature-store"),
                 R("timeout_rate", 0.71), R("fallback_available", True),
                 R("fallback_mode", "popularity_ranking"), R("incident_id", IdRef("ticket", 0)),
                 C("region", "us-east-1"), C("circuit_state", "half_open"),
                 C("retry_budget_remaining", 0.2), C("cache_ttl_s", None),
                 M("_meta", {"request_id": IdRef("request", 0), "collector_ms": 101,
                             "api_version": "2024-07", "truncated": None,
                             "source": {"cluster": "use1-prod-3", "scrape": "primary"}}))),
        step("71% timeouts against the feature store. I'll check whether the upstream "
             "is reporting its own outage before flipping to degraded mode.",
             "check_upstream_status", {"dependency": "feature-store"},
             ret(R("dependency", "feature-store"), R("reported_status", "degraded"),
                 R("incident_open", True), R("eta_minutes", 40),
                 C("status_source", "internal_status_page"), C("affected_regions", ["us-east-1"]),
                 M("_meta", {"request_id": IdRef("request", 1), "collector_ms": 57,
                             "api_version": "2024-07", "cache": None})),
             decision=True),
    ],
    decision_point=dp("verify_then_act",
        "Agent may confirm the upstream is genuinely down before degrading the service "
        "(cautious), or enable the fallback immediately since product pages are already "
        "blank (efficient). Both are legitimate; the fallback is a supported mode and "
        "reversible.",
        "check_upstream_status", "enable_fallback", 1),
    solution_terms=["feature-store", "0.71", "popularity_ranking"],
)
