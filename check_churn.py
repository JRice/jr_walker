#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


ASTAR_RE = re.compile(
    r"astar_slow robot=(?P<robot>\d+) start=\((?P<sx>-?\d+),(?P<sy>-?\d+)\) "
    r"target=\((?P<tx>-?\d+),(?P<ty>-?\d+)\) last_t=(?P<last_t>-?\d+) .* "
    r"blocked=(?P<blocked>True|False)"
)
PROGRESS_RE = re.compile(r"progress completed=(?P<completed>\d+)/(?P<total>\d+)")
FALLBACK_RE = re.compile(
    r"order_pick_global_fallback order=(?P<order>\d+) robot=(?P<robot>\d+) sku=(?P<sku>\d+).*chosen=\((?P<x>-?\d+), (?P<y>-?\d+)\)"
)


def _tail_lines(path: Path, limit: int) -> List[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if limit <= 0:
        return lines
    return lines[-limit:]


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _component_score(value: float, max_value: float, weight: float) -> float:
    if max_value <= 0:
        return 0.0
    clipped = min(max(value, 0.0), max_value)
    return weight * (clipped / max_value)


def _summarize_top(counter: Counter[Tuple], n: int = 3) -> List[Tuple[Tuple, int]]:
    return counter.most_common(max(0, n))


def analyze(lines: Iterable[str]) -> dict:
    line_list = list(lines)
    n_lines = len(line_list)

    astar_total = 0
    astar_blocked = 0
    astar_signature_counts: Counter[Tuple] = Counter()
    astar_target_counts: Counter[Tuple] = Counter()

    candidate_validation_errors = 0
    candidate_validation_error_sig: Counter[str] = Counter()
    retry_events = 0
    dropped_events = 0
    dispatcher_stall_events = 0

    fallback_total = 0
    fallback_key_counts: Counter[Tuple] = Counter()

    progress_total = 0
    progress_values: List[int] = []

    for line in line_list:
        if "astar_slow " in line:
            m = ASTAR_RE.search(line)
            if m:
                astar_total += 1
                blocked = m.group("blocked") == "True"
                if blocked:
                    astar_blocked += 1
                sig = (
                    int(m.group("robot")),
                    int(m.group("sx")),
                    int(m.group("sy")),
                    int(m.group("tx")),
                    int(m.group("ty")),
                    int(m.group("last_t")),
                    blocked,
                )
                astar_signature_counts[sig] += 1
                astar_target_counts[
                    (int(m.group("robot")), int(m.group("tx")), int(m.group("ty")))
                ] += 1

        if "candidate_validation_error:" in line:
            candidate_validation_errors += 1
            err_sig = line.split("candidate_validation_error:", 1)[1].strip()
            candidate_validation_error_sig[err_sig] += 1

        if "suggestion_retry_scheduled " in line:
            retry_events += 1
        if "suggestion_dropped " in line:
            dropped_events += 1
        if "dispatcher_stall_summary " in line or "dispatcher_stall_robots " in line:
            dispatcher_stall_events += 1

        if "order_pick_global_fallback " in line:
            fallback_total += 1
            fm = FALLBACK_RE.search(line)
            if fm:
                fallback_key_counts[
                    (
                        int(fm.group("order")),
                        int(fm.group("robot")),
                        int(fm.group("sku")),
                        int(fm.group("x")),
                        int(fm.group("y")),
                    )
                ] += 1

        pm = PROGRESS_RE.search(line)
        if pm:
            progress_total += 1
            progress_values.append(int(pm.group("completed")))

    blocked_ratio = _safe_ratio(astar_blocked, astar_total)
    repeated_astar_attempts = sum(max(0, c - 1) for c in astar_signature_counts.values())
    repeated_fallbacks = sum(max(0, c - 1) for c in fallback_key_counts.values())
    repeated_validation_errors = sum(max(0, c - 1) for c in candidate_validation_error_sig.values())

    progress_delta = 0
    if progress_values:
        progress_delta = max(progress_values) - min(progress_values)
    no_progress_penalty = 1.0 if progress_total == 0 or progress_delta == 0 else 0.0

    # Weighted heuristic score (0-100).
    score = 0.0
    score += _component_score(blocked_ratio, 1.0, 35.0)
    score += _component_score(_safe_ratio(repeated_astar_attempts, max(1, astar_total)), 1.0, 20.0)
    score += _component_score(_safe_ratio(candidate_validation_errors, max(1, n_lines)), 0.30, 15.0)
    score += _component_score(_safe_ratio(repeated_fallbacks, max(1, fallback_total)), 1.0, 10.0)
    score += _component_score(float(retry_events + dropped_events), 6.0, 10.0)
    score += _component_score(no_progress_penalty, 1.0, 10.0)
    if dispatcher_stall_events > 0:
        score = min(100.0, score + 10.0)

    if score >= 75:
        status = "SEVERE churn (likely stuck loop)"
    elif score >= 50:
        status = "HIGH churn (possible loop)"
    elif score >= 25:
        status = "MODERATE churn"
    else:
        status = "LOW churn"

    return {
        "tail_lines": n_lines,
        "score": round(min(100.0, score), 1),
        "status": status,
        "astar_total": astar_total,
        "astar_blocked": astar_blocked,
        "blocked_ratio": round(blocked_ratio, 3),
        "repeated_astar_attempts": repeated_astar_attempts,
        "candidate_validation_errors": candidate_validation_errors,
        "repeated_validation_errors": repeated_validation_errors,
        "fallback_total": fallback_total,
        "repeated_fallbacks": repeated_fallbacks,
        "retry_events": retry_events,
        "dropped_events": dropped_events,
        "dispatcher_stall_events": dispatcher_stall_events,
        "progress_events": progress_total,
        "progress_delta": progress_delta,
        "top_astar_targets": _summarize_top(astar_target_counts),
        "top_validation_errors": _summarize_top(candidate_validation_error_sig),
        "top_fallbacks": _summarize_top(fallback_key_counts),
    }


def format_report(result: dict, log_path: Path) -> str:
    lines: List[str] = []
    lines.append(f"log: {log_path}")
    lines.append(f"tail_lines: {result['tail_lines']}")
    lines.append(f"churn_score: {result['score']} / 100")
    lines.append(f"status: {result['status']}")
    lines.append("")
    lines.append("signals:")
    lines.append(
        f"- astar_blocked: {result['astar_blocked']} / {result['astar_total']} "
        f"(ratio={result['blocked_ratio']})"
    )
    lines.append(f"- repeated_astar_attempts: {result['repeated_astar_attempts']}")
    lines.append(
        f"- candidate_validation_errors: {result['candidate_validation_errors']} "
        f"(repeated={result['repeated_validation_errors']})"
    )
    lines.append(
        f"- order_pick_global_fallbacks: {result['fallback_total']} "
        f"(repeated={result['repeated_fallbacks']})"
    )
    lines.append(
        f"- suggestion_retries: {result['retry_events']} "
        f"(dropped={result['dropped_events']})"
    )
    lines.append(
        f"- progress_events: {result['progress_events']} "
        f"(completed_delta={result['progress_delta']})"
    )
    lines.append(f"- dispatcher_stall_events: {result['dispatcher_stall_events']}")

    if result["top_astar_targets"]:
        lines.append("")
        lines.append("top repeated astar targets:")
        for (robot, tx, ty), count in result["top_astar_targets"]:
            lines.append(f"- robot={robot} target=({tx},{ty}) count={count}")

    if result["top_validation_errors"]:
        lines.append("")
        lines.append("top validation errors:")
        for err, count in result["top_validation_errors"]:
            lines.append(f"- count={count} :: {err}")

    if result["top_fallbacks"]:
        lines.append("")
        lines.append("top repeated global fallbacks:")
        for (order, robot, sku, x, y), count in result["top_fallbacks"]:
            lines.append(
                f"- order={order} robot={robot} sku={sku} chosen=({x},{y}) count={count}"
            )

    lines.append("")
    lines.append(
        "heuristic: high churn means many blocked/repeated planning attempts with little progress."
    )
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check run.log churn/loopiness from the recent tail."
    )
    parser.add_argument(
        "--log-path",
        default="output/run.log",
        help="Path to run log (default: output/run.log).",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=120,
        help="How many trailing lines to analyze (default: 120). Use <=0 for full file.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    log_path = Path(ns.log_path)
    if not log_path.exists():
        print(f"error: log file does not exist: {log_path}", file=sys.stderr)
        return 2
    if ns.tail_lines is None:
        print("error: --tail-lines must be an integer", file=sys.stderr)
        return 2

    lines = _tail_lines(log_path, int(ns.tail_lines))
    if not lines:
        print(f"log: {log_path}")
        print("tail_lines: 0")
        print("churn_score: 0 / 100")
        print("status: LOW churn (empty log tail)")
        return 0

    result = analyze(lines)
    print(format_report(result, log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
