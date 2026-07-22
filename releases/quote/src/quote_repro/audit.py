from __future__ import annotations

from typing import Any

from .common import candidate_ids, load_json, mean_bool, read_jsonl, release_root, sha256_file, verify_manifest


EXPECTED_CSV_SHA256 = "c38db7b63313960577c6b214f3bdb8979d126532b7afe5ee1f853f6cdf2ae01a"


def audit() -> dict[str, Any]:
    root = release_root()
    integrity = verify_manifest(root)
    csv_path = root / "data" / "quote_circuit_nodes.csv"
    sites = candidate_ids(csv_path)
    if len(sites) != 64 or sha256_file(csv_path) != EXPECTED_CSV_SHA256:
        raise AssertionError("quote candidate universe differs from the certified all-64 universe")

    strict = load_json(root / "artifacts" / "expected" / "strict_full64_result.json")
    selected = strict["soft_runs"]["quote"]["calibrated_soft_handles_behavior"]["raw_cosine_uot"]
    if selected["site_ids"] != ["0.mlp.resid_delta:460"]:
        raise AssertionError("strict PLOT no longer selects the certified quote handle")
    records = read_jsonl(root / "artifacts" / "expected" / "q1_heldout_records.jsonl")
    same = [row for row in records if row["relation"] == "same_u"]
    opposite = [row for row in records if row["relation"] == "opposite_u"]
    wrong = [row for row in records if str(row["relation"]).startswith("wrong_")]
    metrics = {
        "same_u_preserve": mean_bool(same, "patched_preserves_base_sign"),
        "opposite_u_flip": mean_bool(opposite, "patched_matches_source_sign"),
        "wrong_variable_preserve": mean_bool(wrong, "patched_preserves_base_sign"),
    }
    if metrics != {"same_u_preserve": 1.0, "opposite_u_flip": 1.0, "wrong_variable_preserve": 0.0}:
        raise AssertionError(f"Q1 heldout audit failed: {metrics}")

    routing = load_json(root / "artifacts" / "expected" / "pointer_routing_summary.json")
    nonquote = load_json(root / "artifacts" / "expected" / "nonquote_copy_summary.json")
    pointer_top1 = float(routing["summary"]["full_head82"]["ALL"]["target_top1_all"])
    code_top5 = float(nonquote["summary"]["code_before_opener::full_head82_y_to_source_v"]["expected_top5"])
    content_top5 = float(nonquote["summary"]["content_after_opener::full_head82_y_to_source_v"]["expected_top5"])
    q2_supported = pointer_top1 >= 0.90 and code_top5 >= 0.90 and content_top5 >= 0.90
    if q2_supported:
        raise AssertionError("Q2 must not be marked supported by these diagnostics")
    return {
        "integrity": integrity,
        "candidate_count": len(sites),
        "candidate_csv_sha256": sha256_file(csv_path),
        "Q1": {"status": "certified", "handle": selected["site_ids"], "metrics": metrics},
        "Q2": {
            "status": "not_certified",
            "pointer_top1": pointer_top1,
            "nonquote_code_top5": code_top5,
            "nonquote_content_top5": content_top5,
        },
    }
