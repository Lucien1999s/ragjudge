import sys, json
from .judge import LLMJudge
from .metrics import RecallAtK
from .evaluate import evaluate, summarize

def main():
    # usage:
    #   python -m ragjudge.cli run --k 5 < samples.json
    argv = sys.argv[1:]
    if not argv or argv[0] != "run":
        print("usage: python -m ragjudge.cli run [--k K] < samples.json", file=sys.stderr)
        sys.exit(2)

    k = 5
    if len(argv) >= 3 and argv[1] == "--k":
        try:
            k = int(argv[2])
        except ValueError:
            print("--k must be int", file=sys.stderr); sys.exit(2)

    data = json.load(sys.stdin)
    judge = LLMJudge()
    metrics = [RecallAtK(k=k)]
    rows = evaluate(data, judge, metrics)
    print(json.dumps({"results": rows, "summary": summarize(rows)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
