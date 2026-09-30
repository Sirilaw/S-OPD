import copy as cp
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from vlmeval.smp.file import load

FAIL_MSG = 'Failed to obtain answer via API.'
JUDGE_PARSE_FAIL_MSG = 'All 5 retries failed.'
MATHVERSE_AUX_VERSION = 'mcqfix_v2'


def normalize_mathverse_mcq_answer(answer):
    """Normalize a single-choice ground-truth label without guessing."""
    text = str(answer).strip().upper()
    match = re.fullmatch(r'[\(\[\{\s]*([A-F])[\)\]\}\s]*', text)
    return match.group(1) if match else None


def is_mathverse_single_choice(line):
    question_type = str(line.get('question_type', '')).strip().lower()
    return question_type in {'multi-choice', 'multiple-choice', 'mcq'} \
        and normalize_mathverse_mcq_answer(line.get('answer')) is not None


def extract_mathverse_mcq_answer(response):
    """Conservatively extract the final A--F option from a model response.

    The parser prioritizes explicit final-answer syntax and the last standalone
    option line. It deliberately rejects ambiguous outputs such as ``A, C``;
    those continue through MathVerse's judge fallback.
    """
    text = str(response).strip()
    if not text or FAIL_MSG in text:
        return None

    # Direct answers, optionally followed by the selected option text.
    direct = re.fullmatch(
        r'(?is)[\s*_`\[\(]*([A-F])[\]\)]?[\s*_`]*(?:[\.:：\-]\s*[^\n]+)?',
        text,
    )
    if direct:
        return direct.group(1).upper()

    candidates = []

    # Explicit boxed answers, including \boxed{\text{C}}.
    for match in re.finditer(
        r'(?is)\\boxed\s*\{\s*(?:\\text\s*\{\s*)?([A-F])\s*\}?\s*\}',
        text,
    ):
        candidates.append((match.start(), match.group(1).upper()))

    # Strong answer cues. Taking the last cue handles self-correction.
    cue_pattern = (
        r'(?is)(?:final\s+answer|correct\s+(?:answer|option|choice)|'
        r'answer|option|choice)\s*(?:is\s*)?(?::|：|=)?\s*'
        r'[*_`\[\(]*([A-F])(?:\b|(?=\s*[\]\)]))'
    )
    for match in re.finditer(cue_pattern, text):
        candidates.append((match.start(), match.group(1).upper()))

    # A final standalone option line is common in Qwen3-VL responses.
    for match in re.finditer(
        r'(?im)^\s*[*_`\[\(]*([A-F])[\]\)]?[*_`]*\s*[\.!。]?\s*$',
        text,
    ):
        candidates.append((match.start(), match.group(1).upper()))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def parse_mathverse_choices(question):
    """Parse ``A: value`` style choices from a MathVerse evaluation prompt."""
    text = str(question)
    if 'Choices:' not in text:
        return {}
    choices_text = text.split('Choices:', 1)[1]
    matches = list(re.finditer(r'(?m)^\s*([A-F])\s*[:：.)]\s*', choices_text))
    choices = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(choices_text)
        choices[match.group(1)] = choices_text[match.end():end].strip()
    return choices


def _mathverse_latex_to_plain(value):
    value = re.sub(r'\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1)/(\2)', value)
    value = re.sub(r'\\sqrt\s*\{([^{}]+)\}', r'sqrt(\1)', value)
    value = re.sub(r'\\text\s*\{([^{}]*)\}', r'\1', value)
    value = re.sub(r'\\boxed\s*\{([^{}]*)\}', r'\1', value)
    return value


def normalize_mathverse_choice_text(value):
    """Normalize common textual and scalar forms used in MathVerse choices."""
    text = unicodedata.normalize('NFKC', str(value)).strip().lower()
    text = re.sub(r'^\s*(?:extracted\s+answer|answer)\s*[:：]\s*', '', text)
    text = re.sub(r'√\s*\{([^{}]+)\}', r'sqrt(\1)', text)
    text = re.sub(r'√\s*([0-9.]+)', r'sqrt(\1)', text)
    text = _mathverse_latex_to_plain(text)
    text = text.replace('−', '-').replace('^\\circ', '').replace('°', '')
    text = re.sub(r'(?:degrees?|厘米|米|cm|mm|meters?|metres?)\b', '', text)
    text = re.sub(r'[\\$`*_{}\[\]\s,，。!]', '', text)
    return text.strip('\'".')


def _mathverse_safe_scalar(value):
    text = normalize_mathverse_choice_text(value)
    if not text or text in {'null', 'none', 'nan'}:
        return None
    if '=' in text and text.count('=') == 1:
        left, right = text.split('=', 1)
        if re.fullmatch(r'[a-zα-ωθ]+', left):
            text = right
    text = re.sub(r'(?<=\d)(?=sqrt)', '*', text)
    text = text.replace('sqrt', 'math.sqrt')
    if not re.fullmatch(r'[0-9.+\-*/()mathsqrt]+', text):
        return None
    try:
        scalar = float(eval(text, {'__builtins__': {}, 'math': math}, {}))
    except (ArithmeticError, NameError, SyntaxError, TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _mathverse_response_candidates(response):
    text = str(response).strip()
    candidates = [text]
    cues = re.findall(
        r'(?is)(?:therefore|thus|final\s+answer|answer\s+is|is)\s*[:：]?\s*([^\n.]+)',
        text,
    )
    if cues:
        candidates.append(cues[-1].strip())
    return list(dict.fromkeys(candidates))


def map_mathverse_response_to_option(response, question):
    """Map a semantic judge answer to one unique MCQ option when possible."""
    choices = parse_mathverse_choices(question)
    if not choices:
        return None
    normalized_choices = {
        key: normalize_mathverse_choice_text(value) for key, value in choices.items()
    }
    scalar_choices = {key: _mathverse_safe_scalar(value) for key, value in choices.items()}
    matched = set()
    for candidate in _mathverse_response_candidates(response):
        normalized = normalize_mathverse_choice_text(candidate)
        direct = re.fullmatch(r'(?:option)?([a-f])', normalized)
        if direct and direct.group(1).upper() in choices:
            matched.add(direct.group(1).upper())
        matched.update(key for key, value in normalized_choices.items() if normalized == value)
        scalar = _mathverse_safe_scalar(candidate)
        if scalar is not None:
            matched.update(
                key for key, value in scalar_choices.items()
                if value is not None and math.isclose(scalar, value, rel_tol=1e-8, abs_tol=1e-8)
            )
    return next(iter(matched)) if len(matched) == 1 else None


def parse_mathverse_judgement(response):
    """Parse a binary MathVerse judgement without discarding explanations."""
    text = str(response).strip()
    if text in {'0', '1'}:
        return int(text)

    # Local/open-source judges often follow the requested decision but return
    # ``Judgement: 1`` plus an explanation instead of a bare digit. Accept
    # common spellings (judgement, judgment, and the prompt's judement typo)
    # and use the final explicit verdict if the response contains revisions.
    matches = re.findall(
        r'(?i)\bjudg?e?ment["\']?\s*(?:(?:is)\s*)?[:=]?\s*["\'`*_]*([01])\b',
        text,
    )
    if matches:
        return int(matches[-1])
    return None


def parse_mathverse_extraction(response):
    """Return a non-empty extracted answer while retaining raw output separately."""
    text = str(response).strip()
    if not text or FAIL_MSG in text:
        return None
    match = re.fullmatch(r'(?is)extracted\s+answer\s*:\s*(.+)', text)
    return match.group(1).strip() if match else text


def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""" # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0
""" # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4]


def build_mathverse_gpt4_extract_prompt(line):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    prediction = str(line['prediction'])
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt


def build_mathverse_mcq_extract_prompt(line):
    """Ask the judge for an option label while giving it the choices needed to map values."""
    return f"""
Extract the final answer selected by the model response for the multiple-choice
question below. Match a numeric, symbolic, or textual final answer to the
corresponding choice. Output exactly one option letter from A to F. If the
response has no unique final answer or matches no offered choice, output null.
Do not output reasoning, Markdown, or any other text.

[Question and Choices]
{line.get('question_for_eval', '')}

[Model Response]
{line['prediction']}

[Option Letter]
"""


def build_mathverse_gpt4_score_prompt(line):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
For the final test item, output exactly one character: 0 or 1. Do not output
"Judgement:", Markdown, reasoning, or an explanation.\n\n
""" # noqa
    question_for_eval = line['question_for_eval']
    extract = line['extract']
    answer = line['answer']
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model_answer] : {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt


def post_check_score(line, prefetch=False):
    ans = str(line['answer']).strip()
    response = str(line['extract']).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False


def MathVerse_auxeval_extract(model, line):
    is_mcq = is_mathverse_single_choice(line)
    if is_mcq:
        option = extract_mathverse_mcq_answer(line.get('prediction', ''))
        if option is not None:
            return dict(
                log_extract='Deterministic MCQ extraction succeeded',
                extract=option,
                extract_source='rule_mcq_prediction',
                extract_judge_responses=[],
            )

    prompt = (
        build_mathverse_mcq_extract_prompt(line)
        if is_mcq
        else build_mathverse_gpt4_extract_prompt(line)
    )
    logs = []
    judge_responses = []
    retry = 5
    for i in range(retry):
        res = model.generate(prompt, temperature=0.0)
        judge_responses.append(res)
        if is_mcq:
            extract = extract_mathverse_mcq_answer(res)
            if extract is None:
                extract = map_mathverse_response_to_option(res, line.get('question_for_eval', ''))
        else:
            extract = parse_mathverse_extraction(res)

        if extract is None:
            logs.append(f'Try {i}: failed to parse extract response.')
            continue
        logs.append(f'Try {i}: Succeed')
        return dict(
            log_extract='\n'.join(logs),
            extract=extract,
            extract_source='judge_mcq' if is_mcq else 'judge',
            extract_judge_responses=judge_responses,
        )
    if is_mcq:
        # A temperature-zero judge commonly returns the same valid semantic
        # answer (for example ``140°`` or ``null``) on every retry.  If that
        # stable answer is not an offered choice, it is an incorrect MCQ answer,
        # not a judge/parser failure.  Recording it explicitly lets scoring
        # remain deterministic and keeps reported failure rates meaningful.
        responses = [
            str(value).strip()
            for value in judge_responses
            if str(value).strip() and FAIL_MSG not in str(value)
        ]
        majority = Counter(responses).most_common(1)
        if majority and majority[0][1] >= 3:
            majority_response, votes = majority[0]
            logs.append(
                f'Stable non-choice judge answer: {votes}/{len(responses)} majority; '
                'scored as an invalid choice.'
            )
            return dict(
                log_extract='\n'.join(logs),
                extract=majority_response,
                extract_source='judge_mcq_invalid_choice_majority',
                extract_judge_responses=judge_responses,
            )
    logs.append(JUDGE_PARSE_FAIL_MSG)
    return dict(
        log_extract='\n'.join(logs),
        extract='',
        extract_source='judge_parse_failure',
        extract_judge_responses=judge_responses,
    )


def MathVerse_auxeval_score(model, line):
    if is_mathverse_single_choice(line):
        answer = normalize_mathverse_mcq_answer(line.get('answer'))
        prediction = extract_mathverse_mcq_answer(line.get('extract', ''))
        if prediction is not None:
            return dict(
                log_score='Deterministic MCQ exact match',
                score=prediction == answer,
                score_source='rule_mcq_exact_match',
                score_judge_responses=[],
            )
        if line.get('extract_source') == 'judge_mcq_invalid_choice_majority':
            return dict(
                log_score='Stable judge extraction did not match any offered option',
                score=False,
                score_source='rule_mcq_invalid_choice',
                score_judge_responses=[],
            )

    prompt = build_mathverse_gpt4_score_prompt(line)
    logs = []
    judge_responses = []
    retry = 5
    if post_check_score(line, prefetch=True):
        return dict(
            log_score='Prefetch succeed',
            score=True,
            score_source='exact_match',
            score_judge_responses=judge_responses,
        )
    for i in range(retry):
        res = model.generate(prompt, temperature=0.0)
        judge_responses.append(res)
        verdict = None if FAIL_MSG in str(res) else parse_mathverse_judgement(res)

        if verdict is None:
            logs.append(f'Try {i}: failed to parse score response.')
            continue
        logs.append(f'Try {i}: Succeed')
        return dict(
            log_score='\n'.join(logs),
            score=verdict == 1,
            score_source='judge',
            score_judge_responses=judge_responses,
        )
    logs.append(JUDGE_PARSE_FAIL_MSG)
    return dict(
        log_score='\n'.join(logs),
        score=False,
        score_source='judge_parse_failure',
        score_judge_responses=judge_responses,
    )


def MathVerse_acc(result_file):
    df = load(result_file)

    df['metadata'] = df['metadata'].apply(lambda x: x.replace("'", '"'))
    df['metadata'] = df['metadata'].apply(json.loads)
    df_metadata = pd.json_normalize(df['metadata'])
    df = pd.concat([df.drop('metadata', axis=1), df_metadata], axis=1)

    subset = list(set(df['problem_version']))

    res = defaultdict(list)
    for p in subset:
        if p != 'Overall':
            sub = df[df['problem_version'] == p]
        else:
            sub = cp.deepcopy(df)
        res['split'].append(p)
        # Overall Acc
        res['Overall'].append(np.mean(sub['score']) * 100)
        # Subject
        subjects = set(df['subject'])
        for k in subjects:
            res[k].append(np.mean(sub[sub['subject'] == k]['score']) * 100)
        # Subfield
        subfields = set(df['subfield'])
        for k in subfields:
            res[k].append(np.mean(sub[sub['subfield'] == k]['score']) * 100)

    return pd.DataFrame(res)
