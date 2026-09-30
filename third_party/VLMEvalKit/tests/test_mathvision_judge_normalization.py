import pytest
from vlmeval.dataset.utils.mathv import (
    MATH_V_auxeval,
    normalize_mathv_judge_answer,
    post_check,
)


class StubJudge:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.temperatures = []

    def generate(self, prompt, temperature):
        assert 'Return only <answer>ANSWER</answer>' in prompt
        self.temperatures.append(temperature)
        return next(self.responses)


@pytest.mark.parametrize(
    ('response', 'expected'),
    [
        ('<answer>60</answer>', '60'),
        ('reasoning\nExtracted answer: **6**', '6'),
        ('work\nFinal Answer: $1.5$.', '1.5'),
        ('The answer is 42.', '42'),
        (r'work\n\boxed{\frac{1}{2}}', r'\frac{1}{2}'),
        ('B', 'B'),
        ('a long explanation\nwith no explicit final-answer marker', None),
    ],
)
def test_normalize_mathvision_judge_answer(response, expected):
    assert normalize_mathv_judge_answer(response) == expected


def test_existing_verbose_open_answer_is_scored_after_normalization():
    line = {
        'answer': '6',
        'choices': '[]',
        'prediction': 'The answer is six.',
        'res': 'I counted all missing bricks.\nExtracted answer: 6',
    }

    assert post_check(line, prefetch=False) is True


def test_future_evaluation_retries_invalid_long_judge_output():
    judge = StubJudge(
        [
            'I will solve the problem again.\nThe result might be six.',
            'analysis\n<answer>6</answer>',
        ]
    )
    line = {
        'answer': '6',
        'choices': '[]',
        'question': 'What is the missing number?',
        'prediction': 'After counting, the final answer is 6.',
    }

    result = MATH_V_auxeval(judge, line)

    assert result['res'] == '6'
    assert 'invalid answer format' in result['log']
    assert judge.temperatures == [0, 0]
