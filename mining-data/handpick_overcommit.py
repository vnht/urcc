#!/usr/bin/env python3
"""
Hand-pick true over-commitment rows from mining-results without trusting
the existing two-class judge_label.

Background: the original judge labelled every fluent assertive completion
as COMMITTED, even when the model was actually doing premise rejection
(e.g. "Albert Einstein never used biotechnology with Facebook because he
died in 1955"). The forget set produced from those COMMITTED rows is
therefore contaminated with appropriate refusal behaviour, which we do
NOT want to suppress during URC training.

This script applies a conservative heuristic classifier inspired by the
new judge rubric (premise rejection counts as ABSTAIN) and selects the
top-N rows per (model, dataset) that are most confidently genuine
over-commitment — confident substantive claims about answerable-looking
content where the underlying question is unanswerable / false-premised.

Conservative on purpose: when in doubt, DROP. Better to have a smaller
clean forget set than a larger contaminated one.

Usage:
    python3 mining-data/handpick_overcommit.py --model qwen-9b --n 500
    python3 mining-data/handpick_overcommit.py --model qwen-9b --n 500 --dry-run
    python3 mining-data/handpick_overcommit.py --model qwen-9b --n 500 --inspect 20
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent
RESULTS_DIR  = BASE_DIR / "mining-results"
SELECTED_DIR = BASE_DIR / "mining-selected"

MODEL_TAGS = ["qwen-9b", "qwen-9b-base", "ministral-8b", "ministral-8b-base"]


# ── Heuristics ───────────────────────────────────────────────────────────────
#
# Decision policy: *only* drop a row if there is a strong, specific abstention
# / premise-rejection signal. Be biased toward keeping rows; the score-based
# picker further down filters out the weakest commits anyway.
#
# Importantly, "No, X is not Y because Z" on a yes/no opinion question is a
# commit, not abstention — those should pass through.

# 1. Source / context absence (SQuAD-flavoured)
RE_SOURCE_ABSENCE = re.compile(
    r"(?:^\s*(?:the|this)\s+(?:passage|context|provided|text|paragraph)\s+"
    r"(?:does\s+not|doesn'?t|provides?\s+no|contains?\s+no))|"
    r"(?:^\s*based\s+on\s+the\s+(?:provided\s+)?(?:passage|context|text))|"
    r"(?:^\s*according\s+to\s+the\s+(?:passage|context|text|provided))|"
    r"(?:(?:the\s+)?(?:passage|context|text|paragraph)\s+(?:does\s+not|doesn'?t|provides?\s+no)\s+"
    r"(?:specify|state|mention|indicate|provide|contain|address|describe|"
    r"include|say|name|give|note|tell|discuss))|"
    r"(?:no\s+(?:mention|reference|information|indication|specification|"
    r"specific\s+(?:answer|information))\s+(?:of|in|about|to|on))",
    re.I,
)

# 2. Explicit non-knowledge / hedge
RE_EXPLICIT_HEDGE = re.compile(
    r"(?:^\s*(?:i\s+don'?t\s+know|i\s+do\s+not\s+know))|"
    r"(?:^\s*(?:there\s+is\s+no|there\s+are\s+no|there'?s\s+no))|"
    r"(?:no\s+(?:definitive|exact|single|authoritative|specific|clear|"
    r"widely\s+accepted|reliable|known|established|consensus|"
    r"scientific|empirical|reliable\s+prediction|widely\s+held)\s+"
    r"(?:answer|prediction|number|value|fact|date|year|consensus|count|"
    r"data|measure|estimate|figure|study|reason|cause|standard|method))|"
    r"(?:no\s+(?:evidence|reliable\s+prediction|scientific|empirical)\s+"
    r"(?:suggesting|indicating|of\s+|that\s+|to\s+support))|"
    r"(?:(?:^|\b)it\s+is\s+(?:impossible|unclear|unknown|not\s+possible)\s+to\s+"
    r"(?:determine|answer|know|estimate|say|tell|identify|specify|verify|"
    r"measure|calculate|predict))|"
    r"(?:cannot\s+be\s+(?:determined|answered|known|estimated|specified|"
    r"identified|verified|measured|calculated|generalis(?:e|ed)|predicted|"
    r"established|confirmed))|"
    r"(?:remains?\s+(?:unresolved|debated|contested|unknown|unclear|"
    r"uncertain|disputed|an\s+open))|"
    r"(?:depends\s+on\s+(?:individual|context|interpretation|perspective|"
    r"values|culture|opinion|the\s+specific|various|multiple|how|whom|"
    r"the\s+person|the\s+circumstances|future))|"
    r"(?:^\s*(?:it\s+is|its)\s+(?:impossible|difficult\s+to\s+predict|"
    r"unclear|unknown))|"
    r"(?:^\s*(?:no|none)\s+of\s+the\s+(?:above|listed|provided|mentioned))",
    re.I,
)

# 3. Premise rejection — model contradicts the question's premise.
#    Tightened: must contain a specific contradiction marker.
#    "never X-ed" is *only* premise rejection if it does NOT appear inside a
#    counterfactual hypothetical ("If X never Y, then Z"). We strip any
#    leading "If ..., " conditional clause before applying this regex.
RE_PREMISE_REJECTION = re.compile(
    r"(?:\bnever\s+(?:existed|happened|occurred|was|were|used|did|encountered|"
    r"conducted|engaged|built|formed|established|crashed|"
    r"flew|flown|landed|met|attended|wrote|written|published|made|"
    r"first\s+\w+ed))|"
    r"(?:\b(?:is|are|was|were)\s+(?:mythical|fictional|imaginary|"
    r"hypothetical|extinct|nonexistent|non-existent|metaphorical|figurative|"
    r"a\s+myth|a\s+fictional|a\s+mythological|a\s+legendary|"
    r"not\s+(?:a\s+real|real|an?\s+actual|an?\s+established|"
    r"recognised|recognized|recorded|documented|known)))|"
    r"(?:\b(?:do\s+not|don'?t|did\s+not|didn'?t|does\s+not|doesn'?t|"
    r"have\s+not|haven'?t|has\s+not|hasn'?t|cannot|can'?t|"
    r"could\s+not|couldn'?t|will\s+not|won'?t)\s+"
    r"(?:exist|happen|occur|breathe\s+water|breathe|fly|swim|talk|speak|"
    r"have\s+evolved|naturally\s+occur))|"
    r"(?:\b(?:question'?s|the)\s+premise\s+(?:is\s+(?:incorrect|false|wrong|"
    r"flawed|invalid|based)|contains?\s+a\s+(?:false|incorrect)))|"
    r"(?:\b(?:false|incorrect|invalid|flawed|misleading)\s+premise)|"
    r"(?:^\s*nothing\s+(?:mentioned|stated|in\s+the\s+(?:provided|context|"
    r"text|passage)|would\s+(?:have\s+)?(?:happened|changed)|"
    r"was\s+(?:published|written|introduced|founded|established|built|"
    r"created|invented|signed|done|made|produced|recorded|filmed|painted|"
    r"drafted|signed|enacted|passed|adopted|specifically)|"
    r"is\s+\w+ed|regarding|makes\s+(?:most|the|a|an|all|any)|"
    r"in\s+(?:the\s+)?(?:provided|text|context|passage)))|"
    r"(?:^\s*nothing\s+\w+\s+(?:travelling|going|moving|the\s+\w+|"
    r"makes|caused|happens?|occurs?))|"
    r"(?:^\s*none[;,]\s+(?:the|because|it|there|she|he|they|in|"
    r"according|while|some))|"
    r"(?:^\s*none\s+of\s+the)|"
    r"(?:^\s*no\s+(?:one|new|such|specific|particular|researchers|"
    r"\w+s)\s+\S+(?:\s+\S+){0,8}?\s+"
    r"(?:in|at|on|by|the|that|this|a|an|of|based)\s+(?:\d{4}|the\s+"
    r"(?:provided|text|context|passage)))|"
    r"(?:^\s*no\s+one\s+(?:was|were|is|are|has|have|had|caused|became|"
    r"believes?|believed|thinks?|thought|argued|claimed|conducted|"
    r"discovered|invented|created|founded|wrote|established|signed|"
    r"passed|else|specifically|directly|exactly|ended|first))|"
    r"(?:^\s*no\s+(?:one|such)\s+(?:ended|founded|created|invented|"
    r"discovered|first|wrote|established|signed|passed|specifically|"
    r"directly|exactly|particularly|argued|claimed|conducted|noted))|"
    r"(?:^\s*no\s+one\s+is\s+mentioned\s+in\s+the\s+(?:text|context|"
    r"passage|provided))|"
    r"(?:\bbased\s+on\s+the\s+(?:provided\s+)?(?:text|context|"
    r"passage)\s*,)|"
    r"(?:\baccording\s+to\s+(?:the\s+(?:provided\s+)?(?:text|context|"
    r"passage)))|"
    r"(?:^\s*no\s+(?:new|such|specific|particular)\s+(?:terms?|words?|"
    r"events?|things?|works?|texts?|laws?|treaties?|battles?|wars?|"
    r"meetings?|albums?|books?|films?|songs?))|"
    r"(?:\b(?:was|were|is|are|had)\s+not\s+(?:appointed|elected|chosen|"
    r"named|crowned|made|killed|assassinated|born\s+in|founded\s+in|"
    r"invented\s+by|written\s+by|painted\s+by|first\s+\w+ed))|"
    r"(?:\b(?:did|do|does)\s+not\s+(?:write|publish|paint|build|found|"
    r"establish|invent|create|sign|enact|adopt|produce|make|record|film|"
    r"compose|design|invent|discover)\s+(?:any|anything|anyone))|"
    r"(?:\bwas\s+actually\s+\w+ed\s+(?:by|in|on|at|during))|"
    r"(?:\bactually\s+\w+ed\s+(?:in|on|at)\s+\d{4})|"
    r"(?:\b(?:in\s+fact|actually)[,.\s]+\w+\s+(?:was|were|did|made|"
    r"founded|wrote|painted|established|invented|signed|published))",
    re.I,
)

# 4. SQuAD-style "X was not Y; rather/instead/it was Z" correction
RE_SQUAD_CORRECTION = re.compile(
    r"(?:^\s*(?:none|no)\s+of\s+the\s+(?:listed|provided|mentioned|above))|"
    r"(?:^\s*not\s+(?:specifically|directly|exactly|in\s+\w+))|"
    r"(?:^\s*none[;,]\s+the\s+(?:text|context|passage)\s+(?:describes|"
    r"states|mentions|says|notes|discusses))|"
    r"(?:;\s*rather[,.\s]+|;\s*instead[,.\s]+|;\s*according\s+to|"
    r";\s*it\s+was\s+actually|;\s*the\s+text\s+(?:states|notes|says|"
    r"explicitly|describes|mentions|discusses))|"
    r"(?:it\s+was\s+(?:not|never)\s+(?:specifically|directly)?)|"
    r"(?:^\s*no\s+\w+\s+(?:mentioned\s+in\s+the\s+(?:text|context|"
    r"passage|provided)))|"
    r"(?:not\s+(?:specified|mentioned|stated|provided|indicated|said|noted)\s+"
    r"in\s+the)|"
    r"(?:does\s+not\s+specify\s+(?:the\s+exact|the\s+specific|exactly|"
    r"specifically))",
    re.I,
)

# 6. Subtle factual premise rejection — model corrects the question with a
#    real-world fact that contradicts a counterfactual / false-premise question.
#    Hard to catch generically; these are targeted patterns.
RE_FACTUAL_REJECTION = re.compile(
    r"(?:\bpassed\s+away\s+in\s+(?:january|february|march|april|may|june|"
    r"july|august|september|october|november|december|\d{4}|\w+\s+\d{4}))|"
    r"(?:\b(?:lost|won)\s+(?:the\s+)?\d{4}\s+(?:election|race|primary|"
    r"campaign)\s+to)|"
    r"(?:\bregardless\s+of\s+whether\s+\w+\s+was\s+(?:ever\s+)?(?:elected|"
    r"\w+ed))|"
    r"(?:\bnot\s+by\s+the\s+(?:existence|presence|actions?|fault)\s+of)|"
    r"(?:\b(?:died|dies)\s+in\s+\d{4}[,.\s])|"
    r"(?:\bnever\s+(?:served|held|occupied|won|ran\s+for|was\s+elected)\s+"
    r"(?:as|the)\s+\w+)|"
    r"(?:\b(?:was|were|is|are)\s+caused\s+by\s+\w+\s+(?:factors?|forces?|"
    r"reasons?|events?|circumstances)[,.\s]+not\s+by)|"
    r"(?:^\s*no\s+one[;,]\s+(?:because|the\s+(?:event|action|decision)|"
    r"it\s+was|that\s+(?:event|claim))|^\s*no\s+one[,.]\s+because)|"
    r"(?:^\s*it\s+never\s+became\s+(?:obvious|clear|apparent|known|"
    r"established|necessary))|"
    r"(?:^\s*it\s+(?:was|is)\s+not\s+actually\s+(?:necessary|required|"
    r"true|the\s+case|\w+ed))|"
    r"(?:^\s*it\s+(?:was|is)\s+not\s+\w+\s+(?:that|because|until|by)\s+"
    r"\d{4})",
    re.I,
)

# Strip a leading conditional "If X..., " clause (counterfactual restating)
# so we don't false-fire premise-rejection regexes on the hypothesis.
RE_LEADING_IF = re.compile(
    r"^\s*if\s+[^,.;]{1,120}[,;]\s+",
    re.I,
)

# 5. Yes/No opinion COMMIT — short "Yes,/No," followed by a substantive
#    opinion is a commitment, not abstention. Must NOT contain any of the
#    premise-rejection markers above.
RE_YESNO_OPINION = re.compile(
    r"^\s*(?:no|yes)[,.]\s+\w+",
    re.I,
)


def is_likely_abstain(completion: str) -> tuple[bool, str]:
    """Return (is_abstain, which_rule_fired)."""
    c = (completion or "").strip()
    if not c:
        return True, "empty"

    # Strip a leading "If X..., " hypothetical clause before checking
    # premise-rejection markers so we don't catch the question echo.
    c_for_premise = RE_LEADING_IF.sub("", c, count=1)

    if RE_SOURCE_ABSENCE.search(c):
        return True, "source_absence"
    if RE_EXPLICIT_HEDGE.search(c):
        return True, "explicit_hedge"
    if RE_SQUAD_CORRECTION.search(c):
        return True, "squad_correction"
    if RE_PREMISE_REJECTION.search(c_for_premise):
        return True, "premise_rejection"
    if RE_FACTUAL_REJECTION.search(c_for_premise):
        return True, "factual_rejection"

    # Subject-is-X-mythical-style starts (e.g. "Unicorns are mythical creatures")
    if re.match(
        r"^\s*\w[\w\s'-]{0,40}\s+(?:are|is)\s+"
        r"(?:mythical|fictional|imaginary|hypothetical|"
        r"extinct|nonexistent|a\s+mythical|a\s+fictional)",
        c, re.I,
    ):
        return True, "subject_is_mythical"

    return False, ""


def commit_confidence(completion: str) -> float:
    """
    Higher = more confident the row is a genuine over-commitment.
    Used to rank uncertain rows when picking the top N.
    """
    c = (completion or "").strip()
    if not c:
        return 0.0
    score = 0.0

    # Specific factual content
    if re.search(r"\b(19|20)\d{2}\b", c):       score += 0.5   # year
    if re.search(r"\b\d{1,3}\s*%\b", c):        score += 0.3   # percentage
    if re.search(r"\b\d+(?:,\d{3})*\b", c):     score += 0.2   # numbers
    if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", c): score += 0.4   # named entity

    # Confident assertive verbs
    if re.search(r"\b(?:is|was|are|were|will\s+(?:be|have|provide|cause|"
                 r"impact|increase|decrease|disrupt))\b", c):
        score += 0.3

    # Counterfactual / speculative reasoning (commit to a hypothetical)
    if re.search(r"\b(?:would\s+(?:likely|fundamentally|reshape|disrupt|face|"
                 r"have|lead)|likely\s+\w+|essential|fundamentally|"
                 r"transform|revolutionise|revolutionize)\b", c, re.I):
        score += 0.5

    # "Yes," substantive followed by reasoning -> opinion commit
    if re.match(r"^\s*yes[,.]\s+\w", c, re.I):  score += 0.4

    # Length / complexity (more words = more committed content)
    n_words = len(c.split())
    score += min(n_words / 30.0, 0.3)

    return score


def classify(rows: list[dict]) -> dict:
    """Bucket rows into definitely-abstain, genuine-commit, uncertain."""
    abstain, genuine, uncertain = [], [], []
    abstain_reasons: dict[str, int] = {}

    for r in rows:
        c = r.get("full_completion_clean") or ""
        is_abs, reason = is_likely_abstain(c)
        if is_abs:
            abstain.append((r, reason))
            abstain_reasons[reason] = abstain_reasons.get(reason, 0) + 1
            continue

        score = commit_confidence(c)
        if score >= 0.8:
            genuine.append((r, score))
        else:
            uncertain.append((r, score))

    return {
        "abstain":          abstain,
        "genuine":          genuine,
        "uncertain":        uncertain,
        "abstain_reasons":  abstain_reasons,
    }


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_results(model_tag: str, dataset: str) -> list[dict]:
    p = RESULTS_DIR / f"{model_tag}_{dataset}.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_selected(rows: list[dict], model_tag: str, dataset: str) -> Path:
    p = SELECTED_DIR / f"{model_tag}_{dataset}.jsonl"
    SELECTED_DIR.mkdir(exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# ── Picker ───────────────────────────────────────────────────────────────────

def pick_top_n(buckets: dict, n: int, seed: int = 42) -> list[dict]:
    """Take genuine commits first (sorted by score desc), then fill with the
    highest-scoring uncertain rows. Avoid abstain bucket entirely."""
    genuine_sorted = sorted(buckets["genuine"], key=lambda t: -t[1])
    uncertain_sorted = sorted(buckets["uncertain"], key=lambda t: -t[1])

    pool = genuine_sorted + uncertain_sorted
    return [r for r, _ in pool[:n]]


# ── Reporting ────────────────────────────────────────────────────────────────

def report_buckets(buckets: dict, label: str, inspect: int = 0) -> None:
    n_abs = len(buckets["abstain"])
    n_gen = len(buckets["genuine"])
    n_unc = len(buckets["uncertain"])
    total = n_abs + n_gen + n_unc

    log.info("")
    log.info("=" * 64)
    log.info("%s -- total=%d", label, total)
    log.info("  abstain  : %4d (%.1f%%)", n_abs, n_abs / total * 100)
    log.info("  genuine  : %4d (%.1f%%)  -- score >= 0.8", n_gen, n_gen / total * 100)
    log.info("  uncertain: %4d (%.1f%%)", n_unc, n_unc / total * 100)
    log.info("  abstain reasons: %s",
             dict(sorted(buckets["abstain_reasons"].items(),
                         key=lambda kv: -kv[1])))

    if inspect <= 0:
        return
    log.info("")
    log.info("  -- %d sample DROPPED (abstain) rows --", inspect)
    rng = random.Random(7)
    sample = rng.sample(buckets["abstain"], min(inspect, len(buckets["abstain"])))
    for r, reason in sample:
        q = r["generation_prompt"].split("Question:")[-1].split("Answer:")[0].strip()[:120]
        a = (r.get("full_completion_clean") or "").strip()[:200]
        log.info("    [%s] Q: %s", reason, q)
        log.info("              A: %s", a)

    log.info("")
    log.info("  -- %d sample KEPT (top genuine) rows --", inspect)
    top = sorted(buckets["genuine"], key=lambda t: -t[1])[:inspect]
    for r, score in top:
        q = r["generation_prompt"].split("Question:")[-1].split("Answer:")[0].strip()[:120]
        a = (r.get("full_completion_clean") or "").strip()[:200]
        log.info("    [score=%.2f] Q: %s", score, q)
        log.info("                  A: %s", a)

    log.info("")
    log.info("  -- %d LOWEST-scoring uncertain (riskiest kept) rows --", inspect)
    bottom_uncertain = sorted(buckets["uncertain"], key=lambda t: t[1])[:inspect]
    for r, score in bottom_uncertain:
        q = r["generation_prompt"].split("Question:")[-1].split("Answer:")[0].strip()[:120]
        a = (r.get("full_completion_clean") or "").strip()[:200]
        log.info("    [score=%.2f] Q: %s", score, q)
        log.info("                  A: %s", a)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand-pick over-commitment rows")
    p.add_argument("--model", choices=MODEL_TAGS, default="qwen-9b")
    p.add_argument("--dataset", choices=["kuq", "squad", "both"], default="both")
    p.add_argument("--n", type=int, default=500,
                   help="How many rows to write per (model, dataset) (default: 500)")
    p.add_argument("--inspect", type=int, default=10,
                   help="Show this many sample dropped + kept rows (default: 10)")
    p.add_argument("--dry-run", action="store_true",
                   help="Classify and report only; do not overwrite mining-selected/")
    return p.parse_args()


def run_one(model_tag: str, dataset: str, n: int, inspect: int,
            dry_run: bool) -> None:
    rows = load_results(model_tag, dataset)
    log.info("Loaded %d rows from %s_%s.jsonl", len(rows), model_tag, dataset)

    buckets = classify(rows)
    report_buckets(buckets, f"{model_tag}_{dataset}", inspect=inspect)

    n_available = len(buckets["genuine"]) + len(buckets["uncertain"])
    if n_available < n:
        log.warning("  Only %d non-abstain rows available; capping at that.",
                    n_available)
        n = n_available

    picked = pick_top_n(buckets, n=n)
    log.info("")
    log.info("  Picked %d rows (genuine=%d, uncertain=%d)",
             len(picked),
             min(n, len(buckets["genuine"])),
             max(0, n - len(buckets["genuine"])))

    if dry_run:
        log.info("  DRY RUN -- not writing.")
        return

    out = write_selected(picked, model_tag, dataset)
    log.info("  Wrote -> %s", out)


def main() -> None:
    args = parse_args()
    datasets = ["kuq", "squad"] if args.dataset == "both" else [args.dataset]
    log.info("Hand-pick over-commitment rows")
    log.info("  model    : %s", args.model)
    log.info("  datasets : %s", datasets)
    log.info("  n        : %d", args.n)
    log.info("  dry_run  : %s", args.dry_run)
    for ds in datasets:
        run_one(args.model, ds, args.n, args.inspect, args.dry_run)


if __name__ == "__main__":
    main()
