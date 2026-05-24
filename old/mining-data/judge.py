"""
Re-exports the canonical judge from evaluation/judge.py.

The implementation lives in evaluation/judge.py; this shim keeps
`from judge import ...` working inside mining-data/ scripts unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.judge import (  # noqa: F401, E402
    JUDGE_MODEL_ID,
    JUDGE_CEREBRAS_ID,
    JUDGE_TEMPLATE,
    build_judge_prompt,
    _parse_judge_response,
    call_judge,
    make_cerebras_client,
    normalise_label,
    COMMIT,
    ABSTAIN,
)
