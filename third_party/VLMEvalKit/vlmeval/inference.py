import argparse
import asyncio
import os
import os.path as osp
import tempfile
import warnings

import torch
import torch.distributed as dist
from tqdm import tqdm

from vlmeval.config import supported_VLM
from vlmeval.smp import dump, get_logger, get_pred_file_format, get_pred_file_path, get_rank_and_world_size, load
from vlmeval.utils import track_progress_rich
from vlmeval.utils.distributed_env import isolated_model_build_environment
from vlmeval.utils.workload_balance import (
    FileTaskQueue,
    balanced_partitions,
    load_output_token_history,
    order_rows_longest_first,
    safe_profile_name,
    write_output_token_history,
)

logger = get_logger(__name__)
FAIL_MSG = 'Failed to obtain answer via API.'


def _env_true(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _atomic_dump(data, path):
    """Write a checkpoint beside its target and atomically replace it."""
    directory = osp.dirname(path) or '.'
    suffix = osp.splitext(path)[1] or '.pkl'
    fd, tmp_path = tempfile.mkstemp(prefix=f'.{osp.basename(path)}.', suffix=suffix, dir=directory)
    os.close(fd)
    try:
        dump(data, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)


def _build_prompt_struct(model, dataset, dataset_name, row):
    if hasattr(dataset, 'force_use_dataset_prompt') and dataset.force_use_dataset_prompt:
        return dataset.build_prompt(row)
    if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
        return model.build_prompt(row, dataset=dataset_name)
    return dataset.build_prompt(row)


def _prediction_text(value):
    if _is_structured_record(value):
        return str(value['prediction'])
    return str(value)


def _measure_output_tokens(model, data_all):
    processor = getattr(model, 'processor', None)
    tokenizer = getattr(processor, 'tokenizer', processor)
    encode = getattr(tokenizer, 'encode', None)
    measured = {}
    for index, value in data_all.items():
        text = _prediction_text(value)
        if callable(encode):
            try:
                measured[index] = max(1, len(encode(text, add_special_tokens=False)))
                continue
            except (TypeError, ValueError):
                pass
        measured[index] = max(1, (len(text) + 3) // 4)
    return measured


def _dynamic_queue_path(work_dir, model_name, dataset_name, world_size):
    safe_model = safe_profile_name(model_name or 'model')
    safe_dataset = safe_profile_name(dataset_name)
    return osp.join(work_dir, f'.{safe_model}_{safe_dataset}_{world_size}.dynamic_queue')


def _dynamic_inflight_count(model, total, world_size):
    configured = int(
        os.environ.get('VLMEVAL_DYNAMIC_INFLIGHT_PER_GPU', getattr(model, 'max_num_seqs', 8)))
    if configured <= 0:
        raise ValueError('VLMEVAL_DYNAMIC_INFLIGHT_PER_GPU must be positive')
    local_ceiling = max(1, (total + world_size - 1) // world_size)
    return min(configured, local_ceiling)


async def _infer_vllm_dynamic(
    model, dataset, dataset_name, data, ordered_positions, out_file, queue_path,
    rank, world_size, verbose=False,
):
    """Keep one AsyncLLM full per rank while claiming work globally."""
    queue = FileTaskQueue(queue_path, len(ordered_positions))
    inflight = _dynamic_inflight_count(model, len(ordered_positions), world_size)
    initial_span = min(len(ordered_positions), inflight * world_size)
    checkpoint_interval = int(os.environ.get('VLMEVAL_DYNAMIC_CHECKPOINT_INTERVAL', 8))
    if checkpoint_interval <= 0:
        raise ValueError('VLMEVAL_DYNAMIC_CHECKPOINT_INTERVAL must be positive')

    local_res = {}
    completed_since_dump = 0
    state_lock = asyncio.Lock()
    progress = tqdm(desc=f'Dynamic infer {dataset_name}, Rank {rank}/{world_size}', unit='case')

    async def worker(slot):
        nonlocal completed_since_dump
        # Fairly stripe the initial longest-first window across engines. Once
        # that request finishes, this slot joins the shared dynamic queue.
        queue_position = rank + slot * world_size
        if queue_position >= initial_span:
            queue_position = queue.claim()
        while True:
            if queue_position is None:
                return
            row_position = ordered_positions[queue_position]
            row = data.iloc[row_position]
            index = row['index']
            struct = _build_prompt_struct(model, dataset, dataset_name, row)
            request_id = f'r{rank}-s{slot}-q{queue_position}-i{index}'
            try:
                response = await model.generate_async(
                    struct, dataset=dataset_name, request_id=request_id)
            except RuntimeError as err:
                if os.environ.get('SKIP_ERR', False) != '1':
                    raise
                warnings.warn(f'{type(err)} {str(err)}', stacklevel=2)
                response = f'Failed to obtain answer: {type(err)} {str(err)}'

            async with state_lock:
                local_res[index] = response
                completed_since_dump += 1
                if verbose:
                    print(response, flush=True)
                progress.update(1)
                if completed_since_dump >= checkpoint_interval:
                    _atomic_dump(local_res, out_file)
                    completed_since_dump = 0
            queue_position = queue.claim()

    try:
        await asyncio.gather(*(worker(slot) for slot in range(inflight)))
        _atomic_dump(local_res, out_file)
    finally:
        progress.close()
    logger.info(
        f'Dynamic queue rank {rank}/{world_size} completed {len(local_res)} samples '
        f'with an in-flight target of {inflight}.')
    return local_res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, retry_failed=True):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])
    # Build str→orig mapping for checkpoint key conversion
    index_str_to_orig = {str(i): i for i in indices}

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(dataset, 'force_use_dataset_prompt') and dataset.force_use_dataset_prompt:
            struct = dataset.build_prompt(item)
        elif hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_checkpoint.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('https://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data_v11 = load(v11_pred)
                ans_map = {str(x): y for x, y in zip(data_v11['index'], data_v11['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if retry_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}
        logger.info(f'Reuse {len(res)} inference results from previous run.')

    structs = [s for i, s in zip(indices, structs) if str(i) not in res]
    indices = [i for i in indices if str(i) not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        str_indices = [str(i) for i in indices]
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=str_indices)

    # Load the full accumulated results (str keys)
    if osp.exists(out_file):
        res = load(out_file)
    # Convert str keys back to original types for caller compatibility
    result = {index_str_to_orig[k]: v for k, v in res.items() if k in index_str_to_orig}
    if index_set is not None:
        result = {k: v for k, v in result.items() if k in index_set}
    return result


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False,
               retry_failed=True):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    # The custom model config owns the vLLM backend selection. VLMEvalKit's
    # legacy ``use_vllm`` CLI argument can remain False even when that config
    # constructs an async vLLM model, so the runner-set queue flag is the
    # authoritative dispatch switch here.
    dynamic_requested = _env_true('VLMEVAL_VLLM_GLOBAL_QUEUE', default=False)
    balanced_sharding = _env_true('VLMEVAL_BALANCED_SHARDING', default=False)
    profile_dir = os.environ.get('VLMEVAL_WORKLOAD_PROFILE_DIR')
    history = load_output_token_history(profile_dir, dataset_name)
    all_rows = dataset.data.to_dict('records')

    if dynamic_requested:
        # Every rank sees the same unfinished rows. A process-safe counter
        # determines ownership only when an AsyncLLM slot becomes available.
        sheet_indices = list(range(len(dataset)))
    elif balanced_sharding:
        partitions, estimated_loads = balanced_partitions(all_rows, world_size, history)
        sheet_indices = partitions[rank]
        if rank == 0:
            rounded_loads = [round(load, 1) for load in estimated_loads]
            logger.info(
                f'Balanced longest-first sharding for {dataset_name}; '
                f'estimated rank loads: {rounded_loads}; history samples: {len(history)}.')
    else:
        sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        if dynamic_requested:
            _atomic_dump({}, out_file)
        else:
            res = {k: res[k] for k in data_indices}
            _atomic_dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # Prevent local backends from inheriting an incomplete outer torchrun
    # rendezvous while constructing one independent model replica per rank.
    with isolated_model_build_environment():
        model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc,
            retry_failed=retry_failed)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    if dynamic_requested:
        if not getattr(model, 'use_vllm_async', False) or not hasattr(model, 'generate_async'):
            raise RuntimeError(
                'Global dynamic queue requires a model wrapper with use_vllm_async=True '
                'and generate_async().')
        ordered_positions = order_rows_longest_first(data.to_dict('records'), history)
        queue_path = _dynamic_queue_path(work_dir, model_name, dataset_name, world_size)
        queue = FileTaskQueue(queue_path, len(ordered_positions))
        if rank == 0:
            inflight = _dynamic_inflight_count(model, len(ordered_positions), world_size)
            initial_span = min(len(ordered_positions), inflight * world_size)
            queue.initialize(start_position=initial_span)
            logger.info(
                f'Global dynamic queue enabled for {dataset_name}: {len(ordered_positions)} '
                f'unfinished samples, {world_size} engines, {inflight} in-flight per engine, '
                f'{initial_span} fairly striped initial samples, history samples: {len(history)}.')
        if world_size > 1:
            dist.barrier()

        loop = getattr(model, '_vlmeval_async_loop', None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            model._vlmeval_async_loop = loop
        loop.run_until_complete(_infer_vllm_dynamic(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            data=data,
            ordered_positions=ordered_positions,
            out_file=out_file,
            queue_path=queue_path,
            rank=rank,
            world_size=world_size,
            verbose=verbose,
        ))
        return model

    # Local vLLM wrappers can expose a true batch API.  The stock loop below
    # calls llm.generate once per sample and otherwise defeats vLLM batching.
    if getattr(model, 'use_vllm', False) and hasattr(model, 'generate_batch'):
        batch_size = int(os.environ.get('VLMEVAL_VLLM_BATCH_SIZE', 8))
        if batch_size < 0:
            raise ValueError('VLMEVAL_VLLM_BATCH_SIZE must be non-negative')
        # A zero window submits all remaining requests for this rank to vLLM
        # in one call. vLLM still limits active sequences with max_num_seqs,
        # but can immediately refill a freed slot instead of waiting for the
        # slowest item in each outer Python chunk.
        if batch_size == 0:
            batch_size = lt
        with tqdm(total=lt, desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}') as pbar:
            for start in range(0, lt, batch_size):
                rows = [data.iloc[i] for i in range(start, min(start + batch_size, lt))]
                structs = [_build_prompt_struct(model, dataset, dataset_name, row) for row in rows]
                responses = model.generate_batch(structs, dataset=dataset_name)
                if len(responses) != len(rows):
                    raise RuntimeError(f'Batch generation returned {len(responses)} results for {len(rows)} inputs')
                for row, response in zip(rows, responses):
                    res[row['index']] = response
                    if verbose:
                        print(response, flush=True)
                dump(res, out_file)
                pbar.update(len(rows))
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        struct = _build_prompt_struct(model, dataset, dataset_name, data.iloc[i])

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# Add for agent evaluation
def _is_structured_record(v):
    return isinstance(v, dict) and 'prediction' in v and 'extra_records' in v


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, retry_failed=True, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    if rank == 0:
        results = {}
        if osp.exists(prev_file):
            results.update(load(prev_file))
        if osp.exists(result_file):
            data = load(result_file)
            results.update({
                k: v for k, v in zip(data['index'], data['prediction'], strict=True)
            })
        # Recover every completed dynamic/static rank shard after an aborted
        # torchrun. The new queue is then built only from genuinely missing
        # samples, so a claimed task is never permanently lost on restart.
        for i in range(world_size):
            shard_file = tmpl.format(i)
            if osp.exists(shard_file):
                results.update(load(shard_file))
                os.remove(shard_file)
        if retry_failed:
            results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
        if results:
            dump(results, prev_file)
    if world_size > 1:
        dist.barrier()
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm,
        retry_failed=retry_failed)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = load(prev_file) if osp.exists(prev_file) else {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        if os.getenv('SPLIT_THINK', False):
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                prediction = [data_all[x]['prediction'] for x in data['index']]
                extra_records = [data_all[x]['extra_records'] for x in data['index']]
                data['extra_records'] = extra_records
            else:
                prediction = [str(data_all[x]) for x in data['index']]

            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)
            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {split_func}')
            tups = [split_func(x) for x in prediction]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            # data['prediction'] = [str(data_all[x]) for x in data['index']]
            # Add for agent evaluation
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                data['prediction'] = [data_all[x]['prediction'] for x in data['index']]
                data['extra_records'] = [data_all[x]['extra_records'] for x in data['index']]
            else:
                data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        _atomic_dump(data, result_file)
        profile_file = write_output_token_history(
            os.environ.get('VLMEVAL_WORKLOAD_PROFILE_DIR'),
            dataset_name,
            _measure_output_tokens(model, data_all),
        )
        if profile_file is not None:
            logger.info(f'Updated workload profile: {profile_file}')
        for i in range(world_size):
            os.remove(tmpl.format(i))
        queue_file = _dynamic_queue_path(work_dir, model_name, dataset_name, world_size)
        if osp.exists(queue_file):
            os.remove(queue_file)
        # Clean up API checkpoint file
        checkpoint_file = f'{work_dir}/{model_name}_{dataset_name}_checkpoint.pkl'
        if osp.exists(checkpoint_file):
            os.remove(checkpoint_file)
        # Clean up PREV file
        if osp.exists(prev_file):
            os.remove(prev_file)
    if world_size > 1:
        dist.barrier()
    return model
