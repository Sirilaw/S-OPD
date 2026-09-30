from vlmeval.dataset.utils.multiple_choice import (
    build_prompt_mmmu_open,
    extract_answer_from_item,
    infer_mmmu_option,
)


class InvalidJudge:

    def __init__(self):
        self.calls = 0

    def generate(self, _prompt):
        self.calls += 1
        return 'not a valid option'


def test_mmmu_final_option_extraction_prefers_last_explicit_answer():
    choices = {'A': 'first', 'B': 'second', 'C': 'third', 'D': 'fourth'}
    prediction = 'Option A looks plausible. After checking, the final answer is **C**.'
    assert infer_mmmu_option(prediction, choices) == 'C'
    assert infer_mmmu_option(r'Result: \boxed{D}', choices) == 'D'
    assert infer_mmmu_option('I cannot distinguish the answer choices.', choices) is False
    unfinished = 'Answer: A is only a tentative guess. ' + ('still reasoning ' * 50)
    assert infer_mmmu_option(unfinished, choices) is False


def test_mmmu_open_judge_prompt_is_binary_and_explicit():
    prompt = build_prompt_mmmu_open('0.5', '1/2')
    assert 'Output exactly A if they are equivalent' in prompt
    assert 'Reference answer: <start>\n0.5' in prompt
    assert 'Candidate answer: <start>\n1/2' in prompt


def test_mmmu_failed_judge_is_deterministic_instead_of_random():
    judge = InvalidJudge()
    item = {
        'question': 'Pick one',
        'question_type': 'multiple-choice',
        'prediction': 'No final answer was produced.',
        'answer': 'A',
        'A': 'x',
        'B': 'y',
    }
    result = extract_answer_from_item(judge, item, dataset_name='MMMU_DEV_VAL')
    assert result == {'opt': 'Z', 'log': 'Judge failed to return a valid option after 3 attempts.'}
    assert judge.calls == 3
