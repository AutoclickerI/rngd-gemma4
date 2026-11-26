#!/usr/bin/env python3
"""Supervise one server running the public performance stream or accuracy cases."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
STAGE2 = HERE.parents[1] / 'scripts/stage2'
sys.path.insert(0, str(STAGE2))
import accuracy
import performance

BUDGET_SCOPE = 'supervisor-start-through-process-group-cleanup; excludes earlier packaging/extraction'
PROMPT_TOKENS = {'256': 256, '1024': 1024, '2048': 2048, 'image': 512, 'audio': 768}


ENVIRONMENT_POLICY = 'isolated-candidate-v1'
# Only execution infrastructure may come from the worker. Computation/runtime
# switches (including ALL GEMMA4_* and FURIOSA_OPT_PROFILE) must be declared.
RUNTIME_ENVIRONMENT = frozenset({'PATH', 'LD_LIBRARY_PATH', 'HOME', 'TMPDIR',
                                 'LANG', 'LC_ALL', 'TZ', 'RNGD_MODEL_DIR'})


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError('Non-finite JSON number')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_candidate_environment(environment) -> dict:
    if not isinstance(environment, dict) or set(environment) - {'FURIOSA_VISIBLE_CHIPS'}:
        raise ValueError('only FURIOSA_VISIBLE_CHIPS may be configured in the public replay')
    chip = environment.get('FURIOSA_VISIBLE_CHIPS', '0')
    if not isinstance(chip, str) or not re.fullmatch(r'[0-9]+', chip):
        raise ValueError('FURIOSA_VISIBLE_CHIPS must name one card index')
    return dict(environment)


def configuration_digest(environment) -> str:
    return json_digest({'environment_policy': ENVIRONMENT_POLICY,
                        'candidate_environment': validate_candidate_environment(environment)})


def evaluation_identity() -> dict:
    files = {f'scripts/stage2/{name}': STAGE2 / name for name in
             ('accuracy.py', 'performance.py', 'scoring.py', 'suites.json', 'performance_text.json')}
    files.update({f'tools/replay/{name}': HERE / name for name in
                  ('replay_stream.py', 'stream_timing.py')})
    return {name: file_digest(path) for name, path in files.items()}


def candidate_digest(server_sha256, configuration_sha256) -> str:
    return json_digest({'server_sha256': server_sha256,
                        'candidate_configuration_sha256': configuration_sha256})


def launch_environment(plan: dict, inherited: dict, address: str) -> dict:
    env = {key: inherited[key] for key in RUNTIME_ENVIRONMENT if key in inherited}
    env.update(validate_candidate_environment(plan.get('candidate_environment', {})))
    env.update(GEMMA4_API_ADDR=address, GEMMA4_API_KEY='')
    return dict(sorted(env.items()))


def validate_launch_environment(plan: dict, env: dict, address: str) -> None:
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) and '\x00' not in v
                                            for k, v in env.items()):
        raise ValueError('Invalid effective candidate environment')
    if env != launch_environment(plan, env, address):
        raise ValueError('Effective candidate environment differs from the isolated plan')


def build_plan(mode: str, concurrency: int, budget_seconds: float, *,
               candidate_environment: dict | None = None, public_selection: list[int] | None = None) -> dict:
    if type(concurrency) is not int or not 1 <= concurrency <= 100:
        raise ValueError('concurrency must be an integer between 1 and 100')
    if type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('budget_seconds must be finite and positive')
    environment = validate_candidate_environment(candidate_environment if candidate_environment is not None else {})
    if mode == 'performance':
        if public_selection is not None:
            raise ValueError('public_selection applies only to public accuracy')
        cases = [performance.read_input(name) for name in performance.INPUTS]
        warmup = [index % len(cases) for index in range(concurrency)]
        measurement = [index % len(cases) for index in range(15 * concurrency)]
    elif mode == 'accuracy':
        cases = accuracy.read_cases()
        if public_selection is not None:
            if (not isinstance(public_selection, list) or not public_selection
                    or any(type(i) is not int or i < 0 for i in public_selection)
                    or len(set(public_selection)) != len(public_selection)):
                raise ValueError('public_selection requires distinct nonnegative occurrence indices')
            count = len(cases)
            cases = [dict(cases[i % count], id=f"{cases[i % count]['id']}.r{i // count + 1:02d}")
                     for i in public_selection]
        warmup, measurement = [], list(range(len(cases)))
    else:
        raise ValueError('mode must be accuracy or performance')
    return {'schema': 2, 'protocol': 'public-replay-v1', 'mode': mode,
            'concurrency': concurrency, 'budget_seconds': budget_seconds,
            'candidate_environment': environment, 'environment_policy': ENVIRONMENT_POLICY,
            'candidate_configuration_sha256': configuration_digest(environment),
            'public_selection': public_selection,
            'cases': cases, 'warmup_order': warmup, 'measurement_order': measurement}


def validate_plan(plan: dict) -> None:
    expected = build_plan(plan.get('mode'), plan.get('concurrency'), plan.get('budget_seconds'),
                          candidate_environment=plan.get('candidate_environment'),
                          public_selection=plan.get('public_selection'))
    for key, value in expected.items():
        if key not in plan or json_digest(plan[key]) != json_digest(value):
            raise ValueError(f'plan.{key} differs from the public replay')
    if 'server_sha256' in plan and (not isinstance(plan['server_sha256'], str)
                                    or not re.fullmatch('[0-9a-f]{64}', plan['server_sha256'])):
        raise ValueError('Invalid plan.server_sha256')


def emit(out: Path, marker: str, record: dict) -> None:
    line = marker + ' ' + json.dumps(record, ensure_ascii=False, allow_nan=False)
    with (out / 'records.jsonl').open('ab+') as stream:
        # Preserve an interrupted line verbatim and separate subsequent evidence.
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b'\n':
                stream.write(b'\n')
        stream.write((line + '\n').encode())
        stream.flush()
    if marker == 'REPLAY_END':   # the runner reads only the final status from stdout
        print(line, flush=True)


def checkpoint_metadata() -> dict:
    root = os.environ.get('RNGD_MODEL_DIR')
    result = {'root': root, 'selected_directory': None, 'model_file': None,
              'scope': 'Path/stat metadata only; weights are not content-hashed.'}
    if root:
        for directory in (Path(root), Path(root) / 'Gemma4-12B-it', Path(root) / 'gemma-4-12b-it-NVFP4'):
            candidate = directory / 'model.safetensors'
            try:
                stat = candidate.stat()
                if candidate.is_file():
                    result.update(selected_directory=str(directory), model_file={
                        'path': str(candidate), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
                    break
            except OSError:
                continue
    return result


def available_address(address: str) -> str:
    """Resolve a fresh loopback port or refuse an occupied explicit address.

    The candidate opens its own listener, so this probe cannot keep the socket
    reserved across exec. The worker probes again immediately before launch.
    """
    host, separator, port_text = address.rpartition(':')
    if not separator or host not in ('127.0.0.1', 'localhost'):
        raise ValueError('Replay address must be 127.0.0.1:PORT or localhost:PORT')
    port = int(port_text)
    if not 0 <= port <= 65535:
        raise ValueError('Replay port must be between 0 and 65535')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as error:
            raise ValueError(f'Replay address is occupied or unavailable: {address}: {error}') from error
        return '127.0.0.1:' + str(probe.getsockname()[1])


def server_log_tail(out: Path, limit: int = 65536) -> dict:
    """Return bounded diagnostic evidence without changing the experiment verdict."""
    record = {'schema': 1, 'path': 'server.log', 'limit_bytes': limit,
              'bytes_total': None, 'tail_bytes': None, 'truncated': None,
              'text': None, 'error': None}
    try:
        with (out / 'server.log').open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            total = stream.tell()
            stream.seek(max(0, total - limit))
            data = stream.read(limit)
        record.update(bytes_total=total, tail_bytes=len(data), truncated=total > len(data),
                      text=data.decode('utf-8', errors='replace'))
    except OSError as error:
        record['error'] = f'{type(error).__name__}: {error}'
    return record


def _run_worker(plan: dict, server: Path, out: Path, public: Path,
                address: str, epoch: float, deadline: float, *, env: dict,
                server_sha256: str, environment_sha256: str) -> int:
    lock = threading.Lock()
    dispatched = threading.Event()
    dispatched.epoch = epoch
    dispatched.set()
    accuracy.BASE_URL = 'http://' + address
    if plan['mode'] == 'performance':
        import stream_timing
        run_one = stream_timing.measure_request
    else:
        run_one = accuracy.run_request
    records = {'warmup': [], 'measurement': []}
    server_process = None

    def require_candidate_alive():
        if server_process is None or server_process.poll() is not None:
            code = None if server_process is None else server_process.returncode
            raise RuntimeError(f'Candidate server is no longer running: {code}')

    def run_request(phase, index, case, body):
        started = time.monotonic()
        remaining = deadline - started
        if remaining <= 0.01:
            raise TimeoutError('Execution deadline reached before dispatch')
        with lock:
            with (out / 'dispatch.jsonl').open('a') as stream:
                stream.write(json.dumps({'phase': phase, 'request_index': index,
                                         'offset_seconds': started - epoch}) + '\n')
        record = run_one(case, body, index, remaining, dispatched)
        if plan['mode'] == 'performance' and record.get('passed'):
            violations = []
            if record['usage']['completion_tokens'] != 64:
                violations.append('Expected exactly 64 completion tokens')
            expected_prompt = PROMPT_TOKENS[case['suite']]
            if record['usage']['prompt_tokens'] != expected_prompt:
                violations.append(f"Expected {expected_prompt} prompt tokens for {case['suite']}; "
                                  f"got {record['usage']['prompt_tokens']}")
            if violations:
                record.update(passed=False, error='; '.join(violations))
        event = {'schema': 1, 'phase': phase, 'record': record, 'state': 'completed',
                 'replay_started_offset_seconds': started - epoch,
                 'replay_completed_offset_seconds': time.monotonic() - epoch}
        with lock:
            emit(out, 'REPLAY_REQUEST', event)
        return record

    def phase(name):
        order = plan[name + '_order']
        require_candidate_alive()
        if not order:
            return True
        cursor = 0
        stopped = False
        with ThreadPoolExecutor(max_workers=plan['concurrency']) as pool:
            pending = {}
            while pending or cursor < len(order):
                while not stopped and cursor < len(order) and len(pending) < plan['concurrency']:
                    if time.monotonic() >= deadline - 0.01:
                        raise TimeoutError('Execution deadline reached before dispatch')
                    case_index = order[cursor]
                    future = pool.submit(run_request, name, cursor + 1,
                                         plan['cases'][case_index], bodies[case_index])
                    pending[future] = cursor + 1
                    cursor += 1
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                require_candidate_alive()
                for future in completed:
                    del pending[future]
                    record = future.result()
                    records[name].append(record)
                    stopped = stopped or (not record['passed'] and plan['public_selection'] is None)
            require_candidate_alive()
            return len(records[name]) == len(order) and all(r['passed'] for r in records[name])

    result = {'status': 'failed', 'error': None}
    try:
        validate_plan(plan)
        validate_launch_environment(plan, env, address)
        if json_digest(env) != environment_sha256:
            raise ValueError('Effective candidate environment changed before launch')
        if not env.get('RNGD_MODEL_DIR'):
            raise ValueError('RNGD_MODEL_DIR must be supplied by the execution environment')
        if file_digest(server) != server_sha256:
            raise ValueError('Server changed before candidate launch')
        bodies = [accuracy.expand(case['request'], public) for case in plan['cases']]
        if time.monotonic() >= deadline:
            raise TimeoutError('Execution deadline reached before server startup')
        # Inherit the worker's session/process group: the parent can kill every
        # HTTP thread and the server even when a request never returns.
        available_address(address)
        with (out / 'server.log').open('w') as log:
            server_process = subprocess.Popen([str(server)], env=env, stdout=log, stderr=subprocess.STDOUT)
        ready_deadline = deadline
        while True:
            if server_process.poll() is not None:
                raise RuntimeError(f'Server exited during startup: {server_process.returncode}')
            remaining = ready_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Server startup deadline reached')
            try:
                accuracy.request('/v1/models', timeout=min(0.2, remaining))
                require_candidate_alive()
                break
            except (OSError, ValueError):
                time.sleep(min(0.05, max(0, ready_deadline - time.monotonic())))
        (out / 'ready.json').write_text(json.dumps({'seconds': time.monotonic() - epoch}) + '\n')
        if not phase('warmup'):
            result['error'] = 'Warmup failed; no measured request was scheduled'
        elif not phase('measurement'):
            result['error'] = 'One or more measured requests failed'
        else:
            result['status'] = 'complete'
    except Exception as error:
        result = {'status': 'timeout' if isinstance(error, TimeoutError) else 'failed',
                  'error': f'{type(error).__name__}: {error}'}
    finally:
        # The parent owns process-group cleanup, including after this worker exits.
        (out / 'worker-result.json').write_text(json.dumps(result) + '\n')
    return 0 if result['status'] == 'complete' else 1


def _read_events(out: Path) -> tuple[list[dict], list[str]]:
    result, errors = [], []
    for index, line in enumerate((out / 'records.jsonl').read_text().splitlines(), 1):
        if not line:
            continue
        try:
            marker, payload = line.split(' ', 1)
            value = json.loads(payload)
            if marker == 'REPLAY_REQUEST':
                result.append(value)
            elif marker != 'REPLAY_START':
                raise ValueError(f'unexpected marker {marker}')
        except (ValueError, TypeError) as error:
            errors.append(f'Damaged records.jsonl line {index}: {error}')
    return result, errors


def _kill_group(process: subprocess.Popen, deadline: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    # Reserve a short grace for shutdown, but never let stuck HTTP threads own
    # the total wall budget. SIGKILL also handles a child surviving worker exit.
    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        pass


def run(plan_path: Path, server: Path, out: Path, public: Path, address='127.0.0.1:0') -> int:
    epoch = time.monotonic()
    plan_bytes = plan_path.read_bytes()
    plan = strict_json(plan_bytes)
    validate_plan(plan)
    if not server.is_file():
        raise ValueError('Explicit server binary does not exist')
    server_sha256 = file_digest(server)
    if plan.get('server_sha256', server_sha256) != server_sha256:
        raise ValueError('Server differs from plan.server_sha256')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'replay-plan.json').write_bytes(plan_bytes)
    budget = plan['budget_seconds']
    deadline = epoch + budget
    # The limit bounds execution plus cleanup. This reserved interval is reported,
    # rather than pretending it is available to the measured requests.
    execution_window = budget - min(0.5, budget * 0.1)
    execution_deadline = epoch + execution_window
    address_error = None
    try:
        address = available_address(address)
    except (OSError, ValueError) as error:
        address_error = str(error)
    env = launch_environment(plan, os.environ, address)
    validate_launch_environment(plan, env, address)
    (out / 'launch-environment.json').write_text(json.dumps(env) + '\n')
    config_digest = configuration_digest(plan.get('candidate_environment', {}))
    emit(out, 'REPLAY_START', {
        'schema': 1, 'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
        'server_sha256': server_sha256, 'mode': plan['mode'], 'concurrency': plan['concurrency'],
        'budget_seconds': budget, 'execution_window_seconds': execution_window,
        'budget_scope': BUDGET_SCOPE, 'address': address,
        'environment_policy': ENVIRONMENT_POLICY,
        'candidate_environment': plan.get('candidate_environment', {}),
        'candidate_configuration_sha256': config_digest,
        'candidate_sha256': candidate_digest(server_sha256, config_digest),
        'effective_environment': env, 'effective_environment_sha256': json_digest(env),
        'evaluation_source_sha256': evaluation_identity(),
        'model_dir': env.get('RNGD_MODEL_DIR'),
        'checkpoint': checkpoint_metadata()})
    process = None
    status, error = 'failed', None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt('Supervisor received SIGTERM')

    signal.signal(signal.SIGTERM, terminate)
    try:
        if address_error is not None:
            raise ValueError(address_error)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                   '--plan', str((out / 'replay-plan.json').resolve()), '--server', str(server.resolve()),
                   '--out', str(out.resolve()), '--public-test', str(public.resolve()),
                   '--address', address, '--epoch', str(epoch), '--deadline', str(execution_deadline),
                   '--launch-environment', str((out / 'launch-environment.json').resolve()),
                   '--expected-server-sha256', server_sha256,
                   '--effective-environment-sha256', json_digest(env)]
        if time.monotonic() >= execution_deadline:
            raise subprocess.TimeoutExpired(command, execution_window)
        process = subprocess.Popen(command, start_new_session=True)
        process.wait(timeout=max(0, execution_deadline - time.monotonic()))
        if (out / 'worker-result.json').exists():
            result = json.loads((out / 'worker-result.json').read_text())
            status, error = result['status'], result['error']
        else:
            error = f'Worker exited without terminal metadata: {process.returncode}'
    except subprocess.TimeoutExpired:
        status = 'timeout'
        error = ('Total execution deadline reached; process group terminated' if process is not None
                 else 'Total execution deadline reached before worker launch')
    except (Exception, KeyboardInterrupt) as err:
        status, error = 'failed', f'{type(err).__name__}: {err}'
    finally:
        if process is not None:
            _kill_group(process, deadline)
        signal.signal(signal.SIGTERM, previous_sigterm)
    execution_cleanup_seconds = time.monotonic() - epoch
    events, damage = _read_events(out)
    observed = {(r['phase'], r['record']['request_index']) for r in events}
    dispatched = {}
    if (out / 'dispatch.jsonl').exists():
        for line in (out / 'dispatch.jsonl').read_text().splitlines():
            try:
                item = json.loads(line)
                dispatched[item['phase'], item['request_index']] = item['offset_seconds']
            except (json.JSONDecodeError, KeyError, TypeError) as err:
                damage.append(f'Damaged dispatch.jsonl record: {err}')
    server_ready_seconds = None
    if (out / 'ready.json').exists():
        try:
            server_ready_seconds = json.loads((out / 'ready.json').read_text())['seconds']
        except (json.JSONDecodeError, KeyError, TypeError) as err:
            damage.append(f'Damaged ready.json: {err}')
    if damage:
        status = 'timeout' if status == 'timeout' else 'failed'
        error = '; '.join(([error] if error else []) + damage)
    for phase in ('warmup', 'measurement'):
        for index, case_index in enumerate(plan[phase + '_order'], 1):
            if (phase, index) in observed:
                continue
            case = plan['cases'][case_index]
            started = dispatched.get((phase, index))
            event = {'schema': 1, 'phase': phase,
                     'record': {'id': case['id'], 'suite': case['suite'], 'request_index': index,
                                'passed': False, 'error': error or 'Expected request did not complete'},
                     'state': 'not_started' if started is None else 'interrupted',
                     'replay_started_offset_seconds': started,
                     'replay_completed_offset_seconds': None}
            emit(out, 'REPLAY_REQUEST', event)
            events.append(event)
    counts = {}
    for phase in ('warmup', 'measurement'):
        selected = [event for event in events if event['phase'] == phase]
        counts[phase] = {'planned': len(plan[phase + '_order']),
                         'completed': sum(event['state'] == 'completed' for event in selected),
                         'passed': sum(event['record']['passed'] is True for event in selected),
                         'failed': sum(event['record']['passed'] is False for event in selected)}
    all_passed = all(event['record']['passed'] for event in events)
    if status == 'complete' and not all_passed:
        status, error = 'failed', 'Expected requests are missing or failed'
    metrics, gates = None, None
    if status == 'complete' and plan['mode'] == 'performance':
        records = [event['record'] for event in events if event['phase'] == 'measurement']
        metrics = performance.summarize(records)['metrics']
        gates = {'output_tokens_64': all(record['usage']['completion_tokens'] == 64 for record in records),
                 'ttft_p90_within_4000ms': metrics['ttft_ms']['p90'] <= 4000,
                 'estimated_tpot_p90_within_100ms': metrics['estimated_tpot_ms']['p90'] <= 100}
        gates['passed'] = all(gates.values())
    tail = server_log_tail(out)
    emit(out, 'SERVER_LOG_TAIL', tail)
    end = {'schema': 1, 'status': status, 'error': error, 'seconds': time.monotonic() - epoch,
           'execution_cleanup_seconds': execution_cleanup_seconds,
           'counts': counts, 'metrics': metrics, 'gates': gates,
           'server_ready_seconds': server_ready_seconds, 'log_damage': damage,
           'diagnostic_errors': [tail['error']] if tail['error'] else []}
    emit(out, 'REPLAY_END', end)
    # Gates are data (see the module docstring); only an incomplete replay is a failure.
    return 0 if status == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--server', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--public-test', type=Path)
    parser.add_argument('--launch-environment', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--expected-server-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--effective-environment-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--address', default='127.0.0.1:0',
                        help='Loopback address; port 0 allocates a fresh port (default)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--epoch', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.plan or not args.server or not args.out:
        parser.error('--plan, --server and --out are required')
    args.public_test = args.public_test or accuracy.CRATE / 'public-test'
    if args.worker:
        return _run_worker(strict_json(args.plan.read_bytes()), args.server, args.out,
                           args.public_test, args.address, args.epoch, args.deadline,
                           env=strict_json(args.launch_environment.read_bytes()),
                           server_sha256=args.expected_server_sha256, environment_sha256=args.effective_environment_sha256)
    return run(args.plan, args.server, args.out, args.public_test, args.address)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f'replay_stream.py: {error}', file=sys.stderr)
        sys.exit(1)
