from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CandidateVariable:
    variable_id: str
    label: str
    node_ids: tuple[str, ...]
    profile_id: str
    stage: str
    role: str = "identity"


@dataclass(frozen=True)
class CandidateCausalModel:
    model_id: str
    label: str
    variables: tuple[CandidateVariable, ...]
    edges: tuple[tuple[str, str], ...]
    notes: str

    @property
    def variable_count(self) -> int:
        return len(self.variables)

    @property
    def native_node_count(self) -> int:
        return len({node_id for var in self.variables for node_id in var.node_ids})

    def variable_ids(self) -> tuple[str, ...]:
        return tuple(var.variable_id for var in self.variables)


def _v(
    variable_id: str,
    label: str,
    node_ids: Sequence[str],
    profile_id: str,
    stage: str,
    role: str = "identity",
) -> CandidateVariable:
    return CandidateVariable(
        variable_id=variable_id,
        label=label,
        node_ids=tuple(node_ids),
        profile_id=profile_id,
        stage=stage,
        role=role,
    )


OPENING = _v(
    "OpeningQuoteType",
    "opening quote detector pair",
    ("0.mlp.post_act:863", "0.mlp.post_act:2790"),
    "OpeningQuoteType",
    "opening",
)
STORED = _v(
    "StoredQuoteType",
    "layer-0 stored quote type",
    ("0.mlp.resid_delta:460",),
    "StoredQuoteType",
    "storage",
)
COPY_VALUE = _v(
    "CopiedQuoteType",
    "attention value copy scalar",
    ("10.attn.v:663",),
    "CopiedQuoteTypeAtFinalPosition",
    "copy",
)
COPY_READ_VALUE = _v(
    "AttentionReadValueCopy",
    "attention read/value copy pair",
    ("10.attn.act_in:460", "10.attn.v:663"),
    "CopiedQuoteTypeAtFinalPosition",
    "copy",
    role="partial",
)
LOGIT_WRITE = _v(
    "ClosingQuoteLogitPreference",
    "attention output quote-preference write",
    ("10.attn.resid_delta:83",),
    "ClosingQuoteLogitPreference",
    "logit",
)
FINAL_OUTPUT = _v(
    "Output",
    "final residual quote-preference channel",
    ("final_resid:83",),
    "Output",
    "output",
    role="observed_output",
)
OUTPUT_PREF = _v(
    "OutputPreference",
    "attention write plus final residual preference",
    ("10.attn.resid_delta:83", "final_resid:83"),
    "ClosingQuoteLogitPreference",
    "logit",
    role="partial",
)
FULL_PATH = _v(
    "QuoteTypePath",
    "stored type plus downstream copy/output path",
    (
        "0.mlp.resid_delta:460",
        "10.attn.act_in:460",
        "10.attn.v:663",
        "10.attn.resid_delta:83",
        "final_resid:83",
    ),
    "FullQuotePath",
    "path",
)
INTERNAL_PATH = _v(
    "InternalQuoteTypePath",
    "stored type through attention copy and logit write",
    (
        "0.mlp.resid_delta:460",
        "10.attn.act_in:460",
        "10.attn.v:663",
        "10.attn.resid_delta:83",
    ),
    "FullQuotePath",
    "path",
)
QUOTE_IDENTITY = _v(
    "QuoteIdentity",
    "opening detector plus stored quote type",
    ("0.mlp.post_act:863", "0.mlp.post_act:2790", "0.mlp.resid_delta:460"),
    "StoredQuoteType",
    "storage",
)
STORED_PLUS_COPY = _v(
    "StoredAndCopiedQuoteType",
    "stored type plus attention read/value copy",
    ("0.mlp.resid_delta:460", "10.attn.act_in:460", "10.attn.v:663"),
    "StoredQuoteType",
    "storage",
)
STORED_PLUS_OUTPUT = _v(
    "StoredAndOutputPreference",
    "stored type plus output preference",
    ("0.mlp.resid_delta:460", "10.attn.resid_delta:83", "final_resid:83"),
    "StoredQuoteType",
    "storage",
)


CANDIDATE_MODELS: tuple[CandidateCausalModel, ...] = (
    CandidateCausalModel(
        model_id="m1_linear_chain_5",
        label="Linear 5-node chain",
        variables=(OPENING, STORED, COPY_VALUE, LOGIT_WRITE, FINAL_OUTPUT),
        edges=(
            ("OpeningQuoteType", "StoredQuoteType"),
            ("StoredQuoteType", "CopiedQuoteType"),
            ("CopiedQuoteType", "ClosingQuoteLogitPreference"),
            ("ClosingQuoteLogitPreference", "Output"),
        ),
        notes="The user-selected simple chain.",
    ),
    CandidateCausalModel(
        model_id="m2_branch_output_strength",
        label="Branch with output-strength node",
        variables=(OPENING, STORED, COPY_VALUE, OUTPUT_PREF, FINAL_OUTPUT),
        edges=(
            ("OpeningQuoteType", "StoredQuoteType"),
            ("StoredQuoteType", "CopiedQuoteType"),
            ("StoredQuoteType", "OutputPreference"),
            ("CopiedQuoteType", "OutputPreference"),
            ("OutputPreference", "Output"),
        ),
        notes="First refinement suggested after copy/output sites looked partial.",
    ),
    CandidateCausalModel(
        model_id="m3_mechanistic_parallel",
        label="Mechanistic branch with separate copy and output contribution",
        variables=(OPENING, STORED, COPY_READ_VALUE, OUTPUT_PREF, FINAL_OUTPUT),
        edges=(
            ("OpeningQuoteType", "StoredQuoteType"),
            ("StoredQuoteType", "AttentionReadValueCopy"),
            ("StoredQuoteType", "OutputPreference"),
            ("AttentionReadValueCopy", "OutputPreference"),
            ("OutputPreference", "Output"),
        ),
        notes="Second graph suggested in discussion: storage plus copy/output branches.",
    ),
    CandidateCausalModel(
        model_id="m4_minimal_storage_3",
        label="Minimal 3-node storage model",
        variables=(OPENING, STORED, FINAL_OUTPUT),
        edges=(("OpeningQuoteType", "StoredQuoteType"), ("StoredQuoteType", "Output")),
        notes="Smallest direct abstraction: opening detector, storage, output.",
    ),
    CandidateCausalModel(
        model_id="m5_supernode_path_3",
        label="3-node quote-path supernode model",
        variables=(OPENING, FULL_PATH, FINAL_OUTPUT),
        edges=(("OpeningQuoteType", "QuoteTypePath"), ("QuoteTypePath", "Output")),
        notes="Small model that treats the downstream circuit as one operational supernode.",
    ),
    CandidateCausalModel(
        model_id="m6_two_supernodes_4",
        label="4-node storage/output supernode model",
        variables=(OPENING, STORED_PLUS_COPY, OUTPUT_PREF, FINAL_OUTPUT),
        edges=(
            ("OpeningQuoteType", "StoredAndCopiedQuoteType"),
            ("StoredAndCopiedQuoteType", "OutputPreference"),
            ("OutputPreference", "Output"),
        ),
        notes="Refinement splitting the full path into storage/copy and output-preference supernodes.",
    ),
    CandidateCausalModel(
        model_id="m7_internal_path_supernode_3",
        label="3-node internal-path supernode model",
        variables=(OPENING, INTERNAL_PATH, FINAL_OUTPUT),
        edges=(("OpeningQuoteType", "InternalQuoteTypePath"), ("InternalQuoteTypePath", "Output")),
        notes="Full internal causal path as one supernode, with output treated as observed behavior.",
    ),
    CandidateCausalModel(
        model_id="m8_minimal_identity_2",
        label="2-node quote-identity model",
        variables=(QUOTE_IDENTITY, FINAL_OUTPUT),
        edges=(("QuoteIdentity", "Output"),),
        notes="Smallest behaviorally sufficient quote-identity abstraction; does not resolve downstream path.",
    ),
)


def candidate_model_by_id(model_id: str) -> CandidateCausalModel:
    for model in CANDIDATE_MODELS:
        if model.model_id == model_id:
            return model
    raise KeyError(model_id)
