# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from concurrent.futures import ThreadPoolExecutor

from verl.utils.reward_score.math_verify import compute_score


def test_math_verify_score_is_consistent_in_reward_worker_thread():
    model_output = r"<think>Count the objects.</think> Final answer: \boxed{100}"

    direct_score = compute_score(model_output, "100")
    with ThreadPoolExecutor(max_workers=1) as executor:
        threaded_score = executor.submit(compute_score, model_output, "100").result(timeout=30)

    assert direct_score == 1.0
    assert threaded_score == direct_score
