from __future__ import annotations

import sys
import json
from typing import Any, Dict, List
from .judge import LLMJudge
from .metrics import RecallAtK
from .evaluate import evaluate, summarize


def main() -> None:
    """
    Command-line entry point for the RAGJudge package.

    This minimal CLI supports evaluating a JSON list of samples via stdin
    using a fixed judge (LLMJudge) and a single retrieval metric (Recall@K).

    Usage
    -----
    $ python -m ragjudge.cli run --k 5 < samples.json

    Input Format
    -------------
    The input must be a JSON array of objects, e.g.:

    [
      {"id": "1", "question": "...", "answer": "...", "reference": "..."},
      {"id": "2", "question": "...", "answer": "...", "reference": "..."}
    ]

    Output
    -------
    A JSON object printed to stdout with two keys:
        - "results": list of per-sample metric and judge outputs.
        - "summary": mean values aggregated over numeric fields.
    """
    argv = sys.argv[1:]

    # Basic command check
    if not argv or argv[0] != "run":
        print(
            "usage: python -m ragjudge.cli run [--k K] < samples.json",
            file=sys.stderr,
        )
        sys.exit(2)

    # Parse --k argument (defaults to 5)
    k = 5
    if len(argv) >= 3 and argv[1] == "--k":
        try:
            k = int(argv[2])
        except ValueError:
            print("--k must be an integer", file=sys.stderr)
            sys.exit(2)

    # Load input samples from stdin
    try:
        data: List[Dict[str, Any]] = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"[ragjudge] Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(2)

    # Initialize components
    judge = LLMJudge()
    metrics = [RecallAtK(k=k)]

    # Run evaluation
    rows = evaluate(data, judge, metrics)
    output = {"results": rows, "summary": summarize(rows)}

    # Pretty-print JSON (UTF-8 safe)
    print(
        json.dumps(output, ensure_ascii=False, indent=2),
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
