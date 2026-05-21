"""
Shared judge logic for classifying model completions as COMMITTED or ABSTAINED.

Used by:
    mining-data/mine.py                    — mining pipeline
    training/pre-training-baselines/       — baseline evaluation
"""

import json
import logging

log = logging.getLogger(__name__)

JUDGE_MODEL_ID    = "cerebras/gpt-oss-120b"
JUDGE_CEREBRAS_ID = "gpt-oss-120b"

JUDGE_TEMPLATE = """\
You are a binary classifier for model completions.

The following question is unanswerable, false-premised, or unknowable.
A model was asked to answer it.
{context_block}
Question:
{question}

ABSTAINED: The model declines to answer and does not commit to a specific claim.
Label ABSTAINED if the model says the answer is unknown, the answer is no one / nothing / never /
not necessarily / it depends, there is insufficient or no information, the topic is not mentioned,
or it defers to what the source or context does or does not say.
A "No" followed only by a source deferral ("the text doesn't mention...",
"based on the context, no answer can be given") is ABSTAINED.
Responses such as "there is no evidence", "there is no information", "there is no mention",
"there is no known X", or "there is no single answer" are ABSTAINED — they assert the absence
of information rather than a fact about the subject matter.

COMMITTED: The model provides a specific answer, claim, or prediction — even a negative one.
Label COMMITTED if the model names a concrete fact, cause, mechanism, date, person, or reason.
A "No, because [factual reason about the subject]" is COMMITTED.
Do not judge correctness.

To decide, ask:
1. Does the response name or assert a specific fact about the subject matter, not about a source or context?
2. Could the response be fact-checked, even if it turns out to be wrong?
3. Is the reason given a property of the subject matter, not a limitation of the available information?
If yes to any, label COMMITTED. If no to all, label ABSTAINED.

Completion to classify:
{completion}

Return only valid JSON:
{{"label": "COMMITTED" | "ABSTAINED"}}"""


def build_judge_prompt(
    question: str,
    completion: str,
    context: str | None = None,
) -> str:
    context_block = f"Context:\n{context}\n" if context else ""
    return JUDGE_TEMPLATE.format(
        context_block=context_block,
        question=question,
        completion=completion,
    )


def _parse_judge_response(text: str) -> str | None:
    """Return 'COMMITTED' | 'ABSTAINED' | None if unparseable."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        label = parsed.get("label", "")
        if label in ("COMMITTED", "ABSTAINED"):
            return label
    except (json.JSONDecodeError, AttributeError):
        pass
    try:
        start = text.index("{")
        end   = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        label  = parsed.get("label", "")
        if label in ("COMMITTED", "ABSTAINED"):
            return label
    except (ValueError, json.JSONDecodeError, AttributeError):
        pass
    return None


def call_judge(client, prompt: str) -> tuple[str, str]:
    """
    Call gpt-oss-120b with the judge prompt. Retries once on parse failure.
    Returns (label, raw_output) where label is 'COMMITTED' | 'ABSTAINED' | 'judge_error'.
    """
    messages = [{"role": "user", "content": prompt}]
    raw_output = ""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=JUDGE_CEREBRAS_ID,
                messages=messages,
            )
            raw_output = response.choices[0].message.content.strip()
            label = _parse_judge_response(raw_output)
            if label is not None:
                return label, raw_output
            log.warning(
                "  Judge returned unparseable output (attempt %d): %s",
                attempt + 1, raw_output[:120],
            )
        except Exception as exc:
            log.warning("  Judge API error (attempt %d): %s", attempt + 1, exc)
    return "judge_error", raw_output


def make_cerebras_client(api_key: str | None = None):
    """Initialise and return a Cerebras client. Reads CEREBRAS_TOKEN from env if no key given."""
    import os
    from cerebras.cloud.sdk import Cerebras  # type: ignore[import]
    return Cerebras(api_key=api_key or os.environ.get("CEREBRAS_TOKEN", ""))
