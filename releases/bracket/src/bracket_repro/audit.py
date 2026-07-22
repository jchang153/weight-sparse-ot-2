from __future__ import annotations

from collections import defaultdict
from typing import Any

from .common import candidate_ids, load_json, mean_bool, read_jsonl, release_root, sha256_file, verify_manifest


EXPECTED_CSV_SHA256 = "4379a582f1d57051e5e8ebbf7e84252c738bd6708da4646e08f7e23d967be547"


def _directional_mean(rows: list[dict[str, Any]], direction: str, key: str) -> float:
    selected = [row for row in rows if row["direction"] == direction]
    return mean_bool(selected, key)


def _directional_value(rows: list[dict[str, Any]], direction: str, key: str) -> float:
    selected = [float(row[key]) for row in rows if row["direction"] == direction]
    return sum(selected) / len(selected)


def audit() -> dict[str, Any]:
    root = release_root()
    integrity = verify_manifest(root)
    csv_path = root / "data" / "bracket_circuit_nodes.csv"
    sites = candidate_ids(csv_path)
    if len(sites) != 133 or sha256_file(csv_path) != EXPECTED_CSV_SHA256:
        raise AssertionError("bracket candidate universe differs from the certified all-133 universe")

    coarse = load_json(root / "artifacts" / "expected" / "coarse_full133_result.json")
    selected = coarse["soft_runs"]["bracket"]["calibrated_soft_handles_behavior"]["raw_cosine_uot"]
    expected_handle = ["final_resid:1079", "7.mlp.post_act:4133", "7.mlp.resid_delta:2041"]
    if selected["site_ids"] != expected_handle:
        raise AssertionError(f"coarse PLOT handle changed: {selected['site_ids']}")
    coarse_metrics = selected["behavior"]

    final = load_json(root / "artifacts" / "expected" / "balanced_final_result.json")
    threshold = float(final["heldout_direct"]["R_mid"]["acceptance"]["threshold"])
    direct = {}
    for target in ("R_mid", "R_late"):
        gates = {key: float(value) for key, value in final["heldout_direct"][target]["summary"]["gates"].items()}
        validated = all(value >= threshold for value in gates.values())
        direct[target] = {"validated": validated, "worst_gate": min(gates.values()), "gates": gates}
    if not direct["R_mid"]["validated"] or direct["R_late"]["validated"]:
        raise AssertionError("balanced direct-handle certification changed")

    records = read_jsonl(root / "artifacts" / "expected" / "balanced_factorial_records.jsonl")
    factorial = {
        "blocked_output_base_one_to_two": _directional_mean(records, "one_to_two", "blocked_output_base"),
        "blocked_output_base_two_to_one": _directional_mean(records, "two_to_one", "blocked_output_base"),
        "mean_CDE_fraction_one_to_two": _directional_value(records, "one_to_two", "CDE_fraction"),
        "mean_CDE_fraction_two_to_one": _directional_value(records, "two_to_one", "CDE_fraction"),
    }
    clusters: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        clusters[str(row["base_content"])][str(row["direction"])].append(float(row["CDE_fraction"]))
    positive_clusters = {
        direction: sum(
            1
            for values in clusters.values()
            if values.get(direction) and sum(values[direction]) / len(values[direction]) > 0.0
        )
        for direction in ("one_to_two", "two_to_one")
    }
    if positive_clusters != {"one_to_two": 24, "two_to_one": 24}:
        raise AssertionError(f"balanced residual cluster audit changed: {positive_clusters}")
    return {
        "integrity": integrity,
        "candidate_count": len(sites),
        "candidate_csv_sha256": sha256_file(csv_path),
        "B0": {"status": "certified", "handle": selected["site_ids"], "metrics": coarse_metrics},
        "R_mid": direct["R_mid"],
        "R_late": direct["R_late"],
        "B1_pure_chain": {"status": "not_certified", "factorial": factorial},
        "B2_bypass": {"status": "not_certified", "positive_content_clusters": positive_clusters},
    }
