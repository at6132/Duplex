"""
Evaluation metrics for duplex LM experiments.

Metrics:
    - Revision Accuracy: does post-update output contain the corrected value?
    - Contradiction Rate: does output still reference the old/wrong value?
    - Update Latency: how many chars after the update point before correction appears?
    - Coherence Score: simple character-level perplexity of the continuation
"""

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SampleMetrics:
    revision_accurate: bool = False
    has_contradiction: bool = False
    update_latency: int = -1  # chars from start of continuation until correct value appears
    continuation_length: int = 0
    task_type: str = ""


@dataclass
class AggregateMetrics:
    revision_accuracy: float = 0.0
    contradiction_rate: float = 0.0
    mean_update_latency: float = 0.0
    n_samples: int = 0
    per_task: dict = field(default_factory=dict)


def evaluate_sample(
    continuation_text: str,
    expected_values: dict[str, str],
) -> SampleMetrics:
    """
    Evaluate a single generated continuation against expected values.

    expected_values should have at least:
        "correct_value": the value that MUST appear after update
        "wrong_value": the value that should NOT appear after update
    """
    correct = str(expected_values.get("correct_value", ""))
    wrong = str(expected_values.get("wrong_value", ""))

    text_lower = continuation_text.lower()
    correct_lower = correct.lower()
    wrong_lower = wrong.lower()

    revision_accurate = correct_lower in text_lower if correct_lower else False

    has_contradiction = wrong_lower in text_lower if wrong_lower else False

    update_latency = -1
    if revision_accurate:
        idx = text_lower.find(correct_lower)
        update_latency = idx if idx >= 0 else -1

    # Check additional correct values if present
    extra = expected_values.get("extra_correct", "")
    if extra:
        extra_lower = extra.lower()
        if extra_lower not in text_lower:
            revision_accurate = False

    return SampleMetrics(
        revision_accurate=revision_accurate,
        has_contradiction=has_contradiction,
        update_latency=update_latency,
        continuation_length=len(continuation_text),
    )


def compute_metrics(
    results: list[tuple[str, dict[str, str], str]],
) -> AggregateMetrics:
    """
    Compute aggregate metrics over a list of (continuation_text, expected_values, task_type).

    Returns:
        AggregateMetrics with overall and per-task breakdowns
    """
    all_metrics: list[SampleMetrics] = []
    per_task: dict[str, list[SampleMetrics]] = defaultdict(list)

    for continuation_text, expected_values, task_type in results:
        m = evaluate_sample(continuation_text, expected_values)
        m.task_type = task_type
        all_metrics.append(m)
        per_task[task_type].append(m)

    n = len(all_metrics)
    if n == 0:
        return AggregateMetrics()

    revision_accuracy = sum(1 for m in all_metrics if m.revision_accurate) / n
    contradiction_rate = sum(1 for m in all_metrics if m.has_contradiction) / n

    latencies = [m.update_latency for m in all_metrics if m.update_latency >= 0]
    mean_latency = sum(latencies) / len(latencies) if latencies else -1.0

    # Per-task breakdown
    task_agg = {}
    for task_type, task_metrics in per_task.items():
        tn = len(task_metrics)
        task_agg[task_type] = {
            "revision_accuracy": sum(1 for m in task_metrics if m.revision_accurate) / tn,
            "contradiction_rate": sum(1 for m in task_metrics if m.has_contradiction) / tn,
            "mean_update_latency": (
                sum(m.update_latency for m in task_metrics if m.update_latency >= 0)
                / max(1, sum(1 for m in task_metrics if m.update_latency >= 0))
            ),
            "n_samples": tn,
        }

    return AggregateMetrics(
        revision_accuracy=revision_accuracy,
        contradiction_rate=contradiction_rate,
        mean_update_latency=mean_latency,
        n_samples=n,
        per_task=task_agg,
    )
