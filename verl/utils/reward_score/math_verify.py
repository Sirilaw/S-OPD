# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

try:
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

logger = logging.getLogger(__name__)


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    """Score an answer with Math-Verify in both main-thread and worker-thread contexts."""
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        # The high-level math_metric helper enables a SIGALRM timeout internally.
        # Reward managers call synchronous scorers in a thread pool, where Python
        # signals are unsupported. Math-Verify explicitly supports threaded use by
        # disabling its signal timeout in parse/verify.
        extracted_golds = [
            parse(
                ground_truth_boxed,
                extraction_config=(LatexExtractionConfig(),),
                parsing_timeout=None,
            )
        ]
        extracted_predictions = [
            parse(
                model_output,
                extraction_config=(ExprExtractionConfig(), LatexExtractionConfig()),
                parsing_timeout=None,
            )
        ]
        return max(
            1.0
            if any(verify(gold, prediction, timeout_seconds=None) for gold in extracted_golds)
            else 0.0
            for prediction in extracted_predictions
        )
    except Exception:
        logger.exception("Math-Verify failed while scoring a response.")
        return timeout_score
