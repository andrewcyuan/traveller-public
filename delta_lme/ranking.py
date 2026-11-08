"""Query-bound job ranking and a source-scoped evidence handoff contract."""

import math

from .checkpoint import canonical_json

WEIGHTS = {"direct": 3, "supporting": 2, "uncertain": 1, "irrelevant": 0}
CRITERIA = {
    "direct": "The history covers the exact application page or workflow requested by the user query, including evidence contradicting a premise of that query.",
    "supporting": "The history provides complementary evidence needed for the requested comparison, transition, identity or procedure.",
    "uncertain": "The history plausibly matches the query but its scope or evidence is incomplete.",
    "irrelevant": "The history concerns an unrelated application, page, workflow or subject.",
}

CONTRACT = """Evidence handoff contract for this experiment:
Keep the original output schema and 20-state limit. ordered_jobs is a fallible
initial shortlist, not evidence or a restriction: every original trajectory
remains searchable. Do not force a minimum number of trajectories.

Make memory_markdown a concise source ledger that the fixed reader can use:
- State each relevant observed fact separately. Cite its trajectory ID and exact
  state index, and include those supporting states in trajectory_spans.
- Give the page or workflow, role/configuration, and temporal scope when they
  change what a fact means. Keep facts from different surfaces separate.
- Quote exact relevant labels, values and messages. Preserve their distinctions;
  do not silently substitute a similar status, adjacent field or nearby row.
- For a comparison, show each requested side. For a set/count, preserve every
  observed requested member and its scope instead of just a convenient subset.
- For a change, pair before/after observations. A noninitial stored state's
  recorded action produced its snapshot; do not reverse that transition.
- For a procedure, distinguish actions actually performed from necessary steps,
  pre-existing conditions, optional detours and the observed outcome.
- Distinguish positive contradiction of a premise from something merely absent
  in one inspected state or not verified. Preserve exact contrary observations
  when present, rather than replacing them with a vague uncertainty statement.
- Mark conflicting observations and incomplete coverage explicitly. Do not
  invent a global nonexistence claim from a partial screen or another workflow.

The ledger contains source observations, not an answer-key guess or an alternate
reader instruction. Do not put a boxed final answer in it. Prefer a few precise
facts to a long speculative narrative. All facts must be grounded in the original
histories; the reader prompt, sampling and grading remain unchanged.
"""


def payload(query, records, criteria=CRITERIA):
    return {
        "model": "typesafe/jev-1.13",
        "state": {
            "retrieval_query": query,
            **{f"candidate_{index}": row["text"] for index, row in enumerate(records)},
        },
        "questions": {
            f"c{index}": {
                "type": "choice",
                "instructions": f"User retrieval query: {query}\nClassify how useful the history in `candidate_{index}` is for answering this user query.",
                "criteria": criteria,
            }
            for index in range(len(records))
        },
    }


def batches(query, records, criteria=CRITERIA):
    batch = []
    for row in records:
        if len(canonical_json(payload(query, [row], criteria)).encode()) > 150000:
            if batch:
                yield batch
                batch = []
            yield [row], "oversized"
            continue
        if batch and (
            len(batch) >= 16
            or len(canonical_json(payload(query, [*batch, row], criteria)).encode())
            > 150000
        ):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def utility(decision):
    probabilities = decision.get("probabilities")
    if isinstance(probabilities, dict) and set(probabilities) == set(WEIGHTS):
        values = [probabilities[label] for label in WEIGHTS]
        if all(
            isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
            for value in values
        ):
            total = sum(values)
            if total > 0:
                return (
                    sum(
                        probabilities[label] * weight
                        for label, weight in WEIGHTS.items()
                    )
                    / total
                )
    return WEIGHTS[decision["choice"]]


def rank(client, query, records, criteria=CRITERIA):
    decisions = {}
    for group in batches(query, records, criteria):
        if isinstance(group, tuple):
            for row in group[0]:
                decisions[row["id"]] = {
                    "choice": "uncertain",
                    "reason": "complete_source_exceeds_payload_limit",
                }
            continue
        response = client.post("alpha/decisions", payload(query, group, criteria))
        assert set(response["answers"]) == {f"c{index}" for index in range(len(group))}
        for index, row in enumerate(group):
            decision = response["answers"][f"c{index}"]
            assert decision["choice"] in WEIGHTS
            decisions[row["id"]] = decision
    assert len(decisions) == len(records)
    ordered = sorted(
        enumerate(records),
        key=lambda pair: (-utility(decisions[pair[1]["id"]]), pair[0]),
    )
    return {
        "ordered_jobs": [
            [row["id"], utility(decisions[row["id"]])] for _, row in ordered
        ],
        "decisions": decisions,
    }
