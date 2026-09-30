import re
from collections import defaultdict

import pandas as pd
import timeout_decorator

from vlmeval.smp import get_logger, load
from vlmeval.utils import can_infer

logger = get_logger(__name__)

try:
    try:
        from latex2sympy2_extended import latex2sympy
    except ImportError:
        from latex2sympy2 import latex2sympy

except Exception as e:
    logger.critical(f'{type(e)}: {e}')
    logger.critical('Please install latex2sympy2-extended by '
                    'running "pip install latex2sympy2-extended"')
    raise e

FAIL_MSG = 'Failed to obtain answer via API.'
MAX_SHORT_JUDGE_ANSWER_CHARS = 128


def _strip_answer_wrappers(answer):
    answer = str(answer).strip()
    previous = None
    while answer != previous:
        previous = answer
        answer = answer.rstrip(' .').strip()
        if len(answer) >= 4 and answer.startswith('**') and answer.endswith('**'):
            answer = answer[2:-2].strip()
        if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {'`', '$'}:
            answer = answer[1:-1].strip()
    return answer.rstrip(' .').strip()


def _last_boxed_answer(text):
    """Return the content of the last balanced ``\\boxed{...}``, if present."""
    starts = list(re.finditer(r'\\boxed\s*\{', text))
    for match in reversed(starts):
        start = match.end()
        depth = 1
        for pos in range(start, len(text)):
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
                if depth == 0:
                    return text[start:pos].strip()
    return None


def normalize_mathv_judge_answer(response, line=None):
    """Normalize an answer-extraction judge response without using the gold answer.

    MathVision's open-ended scorer compares the extracted answer directly with
    the reference answer.  A judge that returns an explanation followed by the
    answer must therefore be reduced to the explicit answer span first.
    Unstructured long responses are rejected rather than silently treated as
    an answer.
    """
    if response is None:
        return None
    text = str(response).strip()
    if not text or FAIL_MSG in text:
        return None

    # Prefer the explicit contract used by the prompt.  Taking the last match
    # also handles a judge that repeats one of the in-context examples.
    tagged = re.findall(r'(?is)<answer>\s*(.*?)\s*</answer>', text)
    if tagged:
        answer = _strip_answer_wrappers(tagged[-1])
        return answer or None

    markers = re.findall(
        r'(?im)^\s*(?:the\s+)?(?:(?:extracted|final)\s+)?answer\s*'
        r'(?::|=|\bis\b)\s*([^\r\n]+)',
        text,
    )
    if markers:
        answer = _strip_answer_wrappers(markers[-1])
        boxed = _last_boxed_answer(answer)
        answer = _strip_answer_wrappers(boxed if boxed is not None else answer)
        return answer or None

    boxed = _last_boxed_answer(text)
    if boxed is not None:
        answer = _strip_answer_wrappers(boxed)
        return answer or None

    # Multiple-choice responses can still be recovered safely from a verbose
    # response because the choice set constrains the answer.  This does not use
    # the gold label.
    if line is not None:
        try:
            raw_choices = eval(line['choices'])
            if len(raw_choices) > 0:
                choice = can_infer(text, list_to_dict(raw_choices))
                if choice:
                    return str(choice)
        except (KeyError, TypeError, ValueError, SyntaxError):
            pass

    # A single short line is already a valid extraction result.  Do not guess
    # from a long, unstructured explanation: future evaluations will retry it,
    # while existing artifacts will be counted as an explicit parser failure.
    if '\n' not in text and len(text) <= MAX_SHORT_JUDGE_ANSWER_CHARS:
        return _strip_answer_wrappers(text) or None
    return None


@timeout_decorator.timeout(30, use_signals=False)
def is_equal(asw: str, gt_asw: str) -> bool:
    if not isinstance(asw, str) or not isinstance(gt_asw, str):
        print('Warning: input is not string')
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except Exception:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except Exception:
        pass
    return False


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
<answer>14</answer>
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
<answer>0.6</answer>
"""

    example_3 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
<answer>1.45</answer>
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
<answer>[2007, 2008]</answer>
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
<answer>B</answer>
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_mathv_gpt4_prompt(line):
    task_description = """
You are an answer extraction engine, not a problem solver.
Extract the final answer stated by the model response.
Do not explain, reason, correct the response, or restate the question.
Return exactly one answer using this format: <answer>ANSWER</answer>.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model response:\n' + prediction + '\n'
    prompt += 'Return only <answer>ANSWER</answer>:'
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}


def post_check(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    if not prefetch:
        normalized = normalize_mathv_judge_answer(response, line=line)
        if normalized is not None:
            response = normalized
    try:
        if len(eval(line['choices'])) > 0:
            ans = line['answer']
            choices = list_to_dict(eval(line['choices']))
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            res = str(response)
            ans = str(ans)
    except ValueError:
        pass

    try:
        if is_equal(res, ans):
            return res if prefetch else True
        else:
            return False
    except Exception as err:
        logger.warning(f'{type(err)}: {err}')
        return False


def MATH_V_auxeval(model, line):
    prompt = build_mathv_gpt4_prompt(line)
    log = ''
    retry = 5
    if post_check(line, prefetch=True):
        res = post_check(line, prefetch=True)
        return dict(log='Prefetch succeed', res=res)
    for i in range(retry):
        res = model.generate(prompt, temperature=0)

        if FAIL_MSG in res:
            log += f'Try {i}: judge API failed.\n'
            continue

        normalized = normalize_mathv_judge_answer(res, line=line)
        if normalized is None:
            log += f'Try {i}: judge returned an invalid answer format.\n'
            continue

        log += f'Try {i}: Succeed'
        return dict(log=log, res=normalized)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')


def MATH_V_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    from tqdm import tqdm
    for i in tqdm(range(lt)):
        item = data.iloc[i]
        cate = item['category']
        tot['Overall'] += 1
        tot[cate] += 1
        if item['log'] == 'Prefetch succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1
        if post_check(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)
    return pd.DataFrame(res).sort_values('Subject', ignore_index=True)
