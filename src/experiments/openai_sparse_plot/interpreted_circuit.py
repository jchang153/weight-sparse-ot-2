from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema import SparseCircuitEdge, SparseCircuitGraph, SparseCircuitNode


@dataclass(frozen=True)
class InterpretedNodeSpec:
    node_id: str
    role: str
    label: str
    evidence: str
    notes: str = ""


PAPER_BACKED_NODE_SPECS: tuple[InterpretedNodeSpec, ...] = (
    InterpretedNodeSpec(
        node_id="0.mlp.post_act:863",
        role="layer0_mlp_double_quote_detector",
        label="double quote detector",
        evidence="figure_label_and_raw_edge",
        notes="Label visible in OpenAI circuit figure; raw export shows it writes to quote-detector/type residual channels.",
    ),
    InterpretedNodeSpec(
        node_id="0.mlp.post_act:2790",
        role="layer0_mlp_single_quote_detector",
        label="single quote detector",
        evidence="figure_label_and_raw_edge",
        notes="Label visible in OpenAI circuit figure; raw export shows it writes strongly to residual channels 460 and 985.",
    ),
    InterpretedNodeSpec(
        node_id="0.mlp.resid_delta:460",
        role="stored_quote_type_residual_write",
        label="quote type classifier residual channel",
        evidence="paper_text_and_raw_edge",
        notes="Paper text names channel 460 as positive on double quote and negative on single quote.",
    ),
    InterpretedNodeSpec(
        node_id="0.mlp.resid_delta:985",
        role="quote_detector_residual_write",
        label="quote detector residual channel",
        evidence="paper_text_and_raw_edge",
        notes="Paper text names channel 985 as positive on both quote types.",
    ),
    InterpretedNodeSpec(
        node_id="10.attn.act_in:460",
        role="attention_value_reads_quote_type",
        label="layer-10 attention input reads quote type",
        evidence="inferred_from_paper_and_raw_edge",
        notes="Raw export edge from 10.attn.act_in:460 to 10.attn.v:663 matches the paper's value-copy step.",
    ),
    InterpretedNodeSpec(
        node_id="10.attn.act_in:985",
        role="attention_key_reads_quote_detector",
        label="layer-10 attention input reads quote detector",
        evidence="inferred_from_paper_and_raw_edge",
        notes="Raw export edge from 10.attn.act_in:985 to 10.attn.k:657 matches the paper's quote-detector-as-key step.",
    ),
    InterpretedNodeSpec(
        node_id="10.attn.act_in:1013",
        role="attention_query_constant_input",
        label="constant final-position query input",
        evidence="inferred_from_raw_edge",
        notes="Raw export has 10.attn.act_in:1013 -> 10.attn.q:657; paper says the last token has a constant positive-valued query.",
    ),
    InterpretedNodeSpec(
        node_id="10.attn.q:657",
        role="attention_query_channel",
        label="head 82 Q channel 1",
        evidence="paper_text_and_raw_edge",
        notes="657 = 82 * d_head(8) + 1; paper describes head 82 using one QK channel.",
    ),
    InterpretedNodeSpec(
        node_id="10.attn.k:657",
        role="attention_key_channel",
        label="head 82 K channel 1",
        evidence="paper_text_and_raw_edge",
        notes="657 = 82 * d_head(8) + 1; raw export receives quote-detector input.",
    ),
    InterpretedNodeSpec(
        node_id="10.attn.v:663",
        role="attention_value_channel",
        label="head 82 value channel",
        evidence="raw_edge_with_blog_component",
        notes="663 = 82 * d_head(8) + 7 in this released artifact; blog/paper only require one value channel.",
    ),
    InterpretedNodeSpec(
        node_id="10.attn.resid_delta:83",
        role="copied_quote_type_residual_write",
        label="attention output writes quote preference channel",
        evidence="inferred_from_raw_edge",
        notes="Raw export has 10.attn.v:663 -> 10.attn.resid_delta:83.",
    ),
    InterpretedNodeSpec(
        node_id="final_resid:83",
        role="closing_quote_logit_preference_site",
        label="final residual quote-preference channel",
        evidence="inferred_from_raw_export",
        notes="Same residual channel as the layer-10 attention write appears in final_resid; output-logit validation is still required.",
    ),
)


CANONICAL_EDGE_ALLOWLIST: tuple[tuple[str, str], ...] = (
    ("0.mlp.post_act:863", "0.mlp.resid_delta:460"),
    ("0.mlp.post_act:863", "0.mlp.resid_delta:985"),
    ("0.mlp.post_act:2790", "0.mlp.resid_delta:460"),
    ("0.mlp.post_act:2790", "0.mlp.resid_delta:985"),
    ("10.attn.act_in:1013", "10.attn.q:657"),
    ("10.attn.act_in:985", "10.attn.k:657"),
    ("10.attn.act_in:460", "10.attn.v:663"),
    ("10.attn.v:663", "10.attn.resid_delta:83"),
)


RAW_EDGE_MOTIF_TO_AUDIT: tuple[tuple[str, str], ...] = (
    ("0.mlp.act_in:205", "0.mlp.post_act:2790"),
    ("0.mlp.act_in:207", "0.mlp.post_act:863"),
    ("0.mlp.post_act:863", "0.mlp.resid_delta:985"),
    ("0.mlp.post_act:2790", "0.mlp.resid_delta:460"),
    ("0.mlp.post_act:2790", "0.mlp.resid_delta:985"),
    ("10.attn.act_in:1013", "10.attn.q:657"),
    ("10.attn.act_in:985", "10.attn.k:657"),
    ("10.attn.act_in:460", "10.attn.v:663"),
    ("10.attn.v:663", "10.attn.resid_delta:83"),
)


def _node_by_id(graph: SparseCircuitGraph) -> dict[str, SparseCircuitNode]:
    return {node.node_id: node for node in graph.nodes}


def _edge_key(edge: SparseCircuitEdge) -> tuple[str, str]:
    return (edge.src, edge.dst)


def build_interpreted_subcircuit(
    graph: SparseCircuitGraph,
    *,
    include_raw_touching_edges: bool = True,
) -> tuple[SparseCircuitGraph, list[dict[str, Any]], list[dict[str, Any]]]:
    node_lookup = _node_by_id(graph)
    missing = [spec.node_id for spec in PAPER_BACKED_NODE_SPECS if spec.node_id not in node_lookup]
    if missing:
        raise ValueError(f"raw graph is missing interpreted node ids: {missing}")

    spec_by_id = {spec.node_id: spec for spec in PAPER_BACKED_NODE_SPECS}
    sub_nodes = []
    node_rows = []
    for spec in PAPER_BACKED_NODE_SPECS:
        raw_node = node_lookup[spec.node_id]
        merged_metadata = dict(raw_node.metadata)
        merged_metadata.update(
            {
                "interpreted_role": spec.role,
                "interpretation_evidence": spec.evidence,
                "interpretation_notes": spec.notes,
            }
        )
        sub_nodes.append(
            SparseCircuitNode(
                node_id=raw_node.node_id,
                layer=raw_node.layer,
                module=raw_node.module,
                node_kind=raw_node.node_kind,
                index=raw_node.index,
                token_position_role=raw_node.token_position_role,
                published_label=spec.label,
                source_key=raw_node.source_key,
                importance=raw_node.importance,
                metadata=merged_metadata,
            )
        )
        row = raw_node.to_dict()
        row.update(asdict(spec))
        node_rows.append(row)

    allow = set(CANONICAL_EDGE_ALLOWLIST)
    interpreted_ids = set(spec_by_id)
    sub_edges = []
    edge_rows = []
    for edge in graph.edges:
        key = _edge_key(edge)
        if key in allow:
            edge_kind = "canonical_interpreted"
        elif include_raw_touching_edges and (edge.src in interpreted_ids or edge.dst in interpreted_ids):
            edge_kind = "raw_touching_interpreted_node"
        else:
            continue
        metadata = dict(edge.metadata)
        metadata["interpretation_edge_kind"] = edge_kind
        if edge_kind == "canonical_interpreted":
            sub_edges.append(
                SparseCircuitEdge(
                    src=edge.src,
                    dst=edge.dst,
                    weight=edge.weight,
                    edge_kind=edge_kind,
                    metadata=metadata,
                )
            )
        row = edge.to_dict()
        row["interpretation_edge_kind"] = edge_kind
        row["in_raw_edge_motif_to_audit"] = key in set(RAW_EDGE_MOTIF_TO_AUDIT)
        edge_rows.append(row)

    sub_graph = SparseCircuitGraph(
        model=graph.model,
        task=graph.task,
        sweep=graph.sweep,
        k=graph.k,
        source_artifact=graph.source_artifact,
        nodes=tuple(sub_nodes),
        edges=tuple(sub_edges),
        metadata={
            "parent_graph_nodes": len(graph.nodes),
            "parent_graph_edges": len(graph.edges),
            "canonical_edge_allowlist_count": len(CANONICAL_EDGE_ALLOWLIST),
            "canonical_edges_found": sum(1 for edge in sub_edges if edge.edge_kind == "canonical_interpreted"),
            "raw_touching_edges_are_report_only": True,
            "interpretation_warning": (
                "This subcircuit combines source-backed paper/blog labels with raw-export IDs. "
                "It is a candidate operational subcircuit until activation and patching tests validate it."
            ),
        },
    )
    sub_graph.validate()
    return sub_graph, node_rows, edge_rows


def write_interpreted_subcircuit(
    graph: SparseCircuitGraph,
    *,
    out_dir: str | Path,
) -> SparseCircuitGraph:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sub_graph, node_rows, edge_rows = build_interpreted_subcircuit(graph)
    sub_graph.write_json(out / "interpreted_string_closing_subcircuit.json")

    with (out / "interpreted_nodes.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "node_id",
            "role",
            "label",
            "evidence",
            "notes",
            "layer",
            "module",
            "node_kind",
            "index",
            "importance",
            "source_key",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in node_rows:
            row = dict(row)
            row["index"] = json.dumps(row.get("index"))
            writer.writerow(row)

    with (out / "interpreted_edges.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "src",
            "dst",
            "weight",
            "edge_kind",
            "interpretation_edge_kind",
            "in_raw_edge_motif_to_audit",
            "metadata",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in edge_rows:
            row = dict(row)
            row["metadata"] = json.dumps(row.get("metadata", {}), sort_keys=True)
            writer.writerow(row)

    canonical = [edge for edge in sub_graph.edges if edge.edge_kind == "canonical_interpreted"]
    touching_rows = [row for row in edge_rows if row["interpretation_edge_kind"] == "raw_touching_interpreted_node"]
    motif_rows = [row for row in edge_rows if row.get("in_raw_edge_motif_to_audit")]
    lines = [
        "# Interpreted String-Closing Subcircuit",
        "",
        "This file separates raw OpenAI artifact facts from our interpretation.",
        "",
        "## Counts",
        "",
        f"- parent raw nodes: `{len(graph.nodes)}`",
        f"- parent raw edges: `{len(graph.edges)}`",
        f"- interpreted nodes: `{len(sub_graph.nodes)}`",
        f"- canonical interpreted raw edges found: `{len(canonical)}`",
        f"- additional raw edges touching interpreted nodes: `{len(touching_rows)}`",
        f"- raw 9-edge motif candidates present in audit table: `{len(motif_rows)}`",
        "",
        "## Nodes",
        "",
    ]
    for spec in PAPER_BACKED_NODE_SPECS:
        node = next(node for node in sub_graph.nodes if node.node_id == spec.node_id)
        lines.append(
            f"- `{node.node_id}`: {spec.label}; role `{spec.role}`; evidence `{spec.evidence}`; importance `{node.importance}`"
        )
    lines.extend(["", "## Canonical Edges", ""])
    for edge in canonical:
        lines.append(f"- `{edge.src}` -> `{edge.dst}` weight `{edge.weight}`")
    lines.extend(["", "## Raw Touching Edges To Audit", ""])
    for row in touching_rows:
        marker = " [motif-candidate]" if row.get("in_raw_edge_motif_to_audit") else ""
        lines.append(f"- `{row['src']}` -> `{row['dst']}` weight `{row['weight']}`{marker}")
    lines.extend(["", "## Candidate Raw 9-Edge Motif", ""])
    lines.append(
        "This motif is not asserted as the final OpenAI 12-node graph; it is the raw-row edge set to test first because it matches the paper/blog algorithmic path most directly."
    )
    for src, dst in RAW_EDGE_MOTIF_TO_AUDIT:
        matching = [row for row in edge_rows if row["src"] == src and row["dst"] == dst]
        if matching:
            lines.append(f"- `{src}` -> `{dst}` weight `{matching[0]['weight']}`")
        else:
            lines.append(f"- `{src}` -> `{dst}` MISSING")
    (out / "interpreted_subcircuit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sub_graph
