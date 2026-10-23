import concurrent.futures
import copy
import json

import pytest
import requests

from predict.utils.json import write_json_atomic
from predict.api_client.batch import OpenRouterBatchClient
from predict.api_client.pool import ProviderPoolConfig, ProviderSpec


def completion(value=True):
    return {"id": "generation", "model": "openai/test", "object": "chat.completion", "created": 1,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({"ok": value}), "reasoning": "summary"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.text = json.dumps(payload)

    def json(self):
        return copy.deepcopy(self.payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def close(self):
        pass


class Server:
    def __init__(self):
        self.headers, self.jobs, self.posts = {}, {}, []
        self.partial_failure = False
        self.uncertain = False

    def post(self, url, *, data, timeout, stream=False):
        assert url.endswith('/api/v1/batches')
        payload = json.loads(data)
        self.posts.append(copy.deepcopy(payload))
        if self.uncertain:
            raise requests.ConnectionError('connection lost after submission')
        batch_id = f'batch-{len(self.posts)}'
        self.jobs[batch_id] = payload
        return Response({'id': batch_id, 'status': 'in_progress'})

    def get(self, url, *, timeout):
        batch_id = url.rsplit('/', 1)[1]
        rows = []
        for i, row in enumerate(self.jobs[batch_id]['requests']):
            failed = self.partial_failure and i == 0
            rows.append({'custom_id': row['custom_id'],
                         'response': {'status_code': 500 if failed else 200, 'body': completion(row['body']['messages'][0]['content'])},
                         'error': {'message': 'upstream failed'} if failed else None})
        return Response({'id': batch_id, 'status': 'completed', 'completion_window': '24h',
                         'results': list(reversed(rows))})

    def close(self):
        pass


def client(tmp_path, monkeypatch, server, **overrides):
    monkeypatch.setattr('predict.api_client.batch.requests.Session', lambda: server)
    options = {'max_requests': 2, 'flush_seconds': .02, 'poll_seconds': .02}
    options.update(overrides.pop('batch_options', {}))
    return OpenRouterBatchClient(
        batch_dir=tmp_path, batch_options=options, api_key='test-secret',
        base_url='https://openrouter.ai/api/v1', model='openai/test', timeout_s=60,
        max_tokens=1024, temperature=None, tool_service_url='http://localhost',
        enable_group_tools=False, max_tool_rounds=0, reasoning_effort='medium',
        enable_thinking=False, transport_max_retries=0, **overrides)


def test_batches_preserve_messages_and_match_out_of_order_results(tmp_path, monkeypatch):
    server = Server()
    c = client(tmp_path, monkeypatch, server)
    messages = [[{'role': 'user', 'content': value}] for value in ('first', 'second')]
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(c.chat_json, messages))
        assert len(server.posts) == 1
        assert [r['content']['ok'] for r in responses] == ['first', 'second']
        assert all(r['reasoning_content'] == 'summary' and r['usage']['total_tokens'] == 30 for r in responses)
        for row in server.posts[0]['requests']:
            assert row['body']['messages'] in messages
            assert row['body']['reasoning_effort'] == 'medium'
            assert 'temperature' not in row['body']
        assert 'test-secret' not in next(tmp_path.glob('job_*.json')).read_text()
    finally:
        c.close()


def test_completed_results_recover_without_paid_resubmission(tmp_path, monkeypatch):
    server = Server()
    c = client(tmp_path, monkeypatch, server)
    messages = [{'role': 'user', 'content': 'resume'}]
    first = c.chat_json(messages)
    c.close()
    resumed = client(tmp_path, monkeypatch, server)
    try:
        second = resumed.chat_json(messages)
        assert second['batch']['recovered'] is True
        assert second['batch']['id'] == first['batch']['id']
        assert len(server.posts) == 1
        # A subsequent intentional call is a distinct request, not a permanent cache.
        resumed.chat_json(messages)
        assert len(server.posts) == 2
    finally:
        resumed.close()


def test_pending_job_is_polled_on_restart(tmp_path, monkeypatch):
    server = Server()
    c = client(tmp_path, monkeypatch, server)
    messages = [{'role': 'user', 'content': 'pending'}]
    c.chat_json(messages)
    c.close()
    path = next(tmp_path.glob('job_*.json'))
    job = json.loads(path.read_text())
    job.update(status='in_progress', remote={'id': 'batch-1', 'status': 'in_progress'}, next_poll_at=0)
    write_json_atomic(path, job)
    resumed = client(tmp_path, monkeypatch, server)
    try:
        assert resumed.chat_json(messages)['content']['ok'] == 'pending'
        assert len(server.posts) == 1
    finally:
        resumed.close()


def test_partial_failure_keeps_successful_sibling_and_retries_only_failed_call(tmp_path, monkeypatch):
    server = Server()
    server.partial_failure = True
    c = client(tmp_path, monkeypatch, server)
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(c.chat_json, [{'role': 'user', 'content': x}]) for x in ('a', 'b')]
            outcomes = [f.exception(timeout=5) for f in futures]
        assert sum(x is None for x in outcomes) == 1
        failed = next(i for i, error in enumerate(outcomes) if error)
        server.partial_failure = False
        c.chat_json([{'role': 'user', 'content': ('a', 'b')[failed]}])
        assert [len(p['requests']) for p in server.posts] == [2, 1]
    finally:
        c.close()


def test_uncertain_submission_blocks_resubmit_and_restart(tmp_path, monkeypatch):
    server = Server()
    server.uncertain = True
    c = client(tmp_path, monkeypatch, server)
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='connection lost'):
                c.chat_json([{'role': 'user', 'content': 'once'}])
        assert len(server.posts) == 1
    finally:
        c.close()
    with pytest.raises(RuntimeError, match='Submission outcome unknown'):
        client(tmp_path, monkeypatch, server)
    assert len(server.posts) == 1


def test_live_journal_cannot_be_opened_twice(tmp_path, monkeypatch):
    server = Server()
    c = client(tmp_path, monkeypatch, server)
    try:
        with pytest.raises(ValueError, match='already in use'):
            client(tmp_path, monkeypatch, server)
    finally:
        c.close()


def test_batch_provider_requires_explicit_no_failover():
    spec = ProviderSpec.from_mapping({'name': 'batch', 'model': 'openai/test',
        'base_url': 'https://openrouter.ai/api/v1', 'api_key_env': 'KEY',
        'max_inflight': 16, 'transport': 'openrouter_batch', 'temperature': None,
        'reasoning_effort': 'medium'})
    with pytest.raises(ValueError, match='automatic failover'):
        ProviderPoolConfig(providers=(spec,)).validate()
    valid = ProviderPoolConfig(providers=(spec,), max_failovers=0)
    valid.validate()
    assert valid.public_dict()['providers'][0]['transport'] == 'openrouter_batch'


@pytest.mark.parametrize('initial_status', [404, 503])
def test_poll_failure_does_not_resubmit_a_paid_job(tmp_path, monkeypatch, initial_status):
    server = Server()
    original_get = server.get
    calls = []
    def flaky_get(url, *, timeout):
        calls.append(url)
        if len(calls) == 1:
            return Response({'error': 'job temporarily unavailable'}, initial_status)
        return original_get(url, timeout=timeout)
    server.get = flaky_get
    c = client(tmp_path, monkeypatch, server)
    try:
        assert c.chat_json([{'role': 'user', 'content': 'poll'}])['content']['ok'] == 'poll'
        assert len(calls) == 2 and len(server.posts) == 1
    finally:
        c.close()


def test_poll_persists_changes_without_rewriting_unchanged_jobs(tmp_path, monkeypatch):
    server = Server()
    c = client(tmp_path, monkeypatch, server)
    remote = {"id": "batch-1", "status": "in_progress", "request_counts": {"total": 2, "completed": 0}}
    job = {"status": "in_progress", "remote": copy.deepcopy(remote), "request": {"requests": []}}
    saved = []
    monkeypatch.setattr(c, "_save", lambda value: saved.append(copy.deepcopy(value)))
    monkeypatch.setattr(server, "get", lambda *args, **kwargs: Response(remote))
    try:
        c._poll(job)
        assert not saved and job["next_poll_at"] > 0
        remote["request_counts"]["completed"] = 1
        c._poll(job)
        assert len(saved) == 1
        monkeypatch.setattr(server, "get", lambda *args, **kwargs: Response({}, 503))
        c._poll(job)
        c._poll(job)
        assert len(saved) == 2 and "last_poll_error" in saved[-1]
        monkeypatch.setattr(server, "get", lambda *args, **kwargs: Response(remote))
        c._poll(job)
        assert len(saved) == 3 and "last_poll_error" not in saved[-1]
        remote.update(status="completed", results=[])
        c._poll(job)
        assert len(saved) == 4 and saved[-1]["status"] == "completed"
        assert not server.posts
    finally:
        c.close()


def test_shared_pool_uses_batch_and_closes_its_worker(tmp_path, monkeypatch):
    from predict.api_client.pool import build_provider_pool
    server = Server()
    monkeypatch.setattr('predict.api_client.batch.requests.Session', lambda: server)
    monkeypatch.setenv('TEST_BATCH_KEY', 'sk-or-test-placeholder-not-a-real-key')
    spec = ProviderSpec(name='batch', base_url='https://openrouter.ai/api/v1', model='openai/test',
        api_key_env='TEST_BATCH_KEY', max_inflight=2, transport='openrouter_batch',
        batch_options={'flush_seconds': .01, 'poll_seconds': .01})
    config = ProviderPoolConfig(providers=(spec,), max_failovers=0)
    pool = build_provider_pool(config, env_file=None, timeout_s=60, max_tokens=1024,
        temperature=None, reasoning_effort='medium', enable_thinking=False,
        tool_service_url='', enable_group_tools=False, max_tool_rounds=0,
        transport_max_retries=0, batch_dir=tmp_path, token_limit_parameter='max_tokens')
    try:
        pool.prefetch_json([{'role':'user','content':'integrated'}])
        result = pool.chat_json([{'role':'user','content':'integrated'}])
        assert result['content']['ok'] == 'integrated'
        assert result['execution_provider']['requested_model'] == 'openai/test'
        body = server.posts[0]['requests'][0]['body']
        assert body['reasoning_effort'] == 'medium' and body['max_tokens'] == 1024
    finally:
        pool.close()
    assert not pool._states[0].client._worker.is_alive()


def test_prefetch_submits_all_before_results_and_recovers_without_duplicates(tmp_path, monkeypatch):
    import threading
    import time
    from predict.api_client.pool import OpenAIProviderPool

    finished = threading.Event()
    class SlowServer(Server):
        def get(self, url, *, timeout):
            if not finished.is_set():
                return Response({'id': url.rsplit('/', 1)[1], 'status': 'in_progress'})
            return super().get(url, timeout=timeout)
    server = SlowServer()
    spec = ProviderSpec(name='batch', base_url='https://openrouter.ai/api/v1', model='openai/test',
                        api_key_env='KEY', max_inflight=1, transport='openrouter_batch')
    config = ProviderPoolConfig(providers=(spec,), max_failovers=0)
    pool = OpenAIProviderPool(config, client_factory=lambda _: client(tmp_path, monkeypatch, server))
    # Identical messages are separate experiment requests, not a global cache.
    messages = [[{'role': 'user', 'content': x}] for x in ['same', 'same', 'a', 'b', 'c']]
    try:
        assert pool.supports_batch_prefetch
        for m in messages:
            pool.prefetch_json(m)
        deadline = time.monotonic()+5
        while sum(len(p['requests']) for p in server.posts) < len(messages):
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert not finished.is_set()
    finally:
        pool.close()  # Leave all remote jobs in flight.
    post_count = len(server.posts)
    resumed = OpenAIProviderPool(config, client_factory=lambda _: client(tmp_path, monkeypatch, server))
    try:
        for m in messages:
            resumed.prefetch_json(m)
        finished.set()
        outputs = [resumed.chat_json(m) for m in messages]
        assert len(server.posts) == post_count
        assert len({o['batch']['custom_id'] for o in outputs}) == 5
        assert all(o['batch']['recovered'] for o in outputs)
        assert [o['content']['ok'] for o in outputs] == ['same', 'same', 'a', 'b', 'c']
    finally:
        resumed.close()


def test_explicit_429_waits_and_requeues_same_call_without_failing_consumer(tmp_path, monkeypatch):
    import time
    class LimitedServer(Server):
        attempts = 0
        def post(self, url, *, data, timeout, stream=False):
            self.attempts += 1
            if self.attempts == 1:
                self.rejected = json.loads(data)
                # A truncated 429 body must still be treated as a known rejection.
                response = Response({}, 429)
                response.headers = {'Retry-After': '60'}
                def broken(): raise requests.exceptions.ChunkedEncodingError('truncated 429 body')
                response.json = broken
                return response
            return super().post(url, data=data, timeout=timeout, stream=stream)
    server = LimitedServer()
    c = client(tmp_path, monkeypatch, server)
    messages = [{'role': 'user', 'content': 'rate-limited'}]
    try:
        c.prefetch_json(messages)
        deadline = time.monotonic()+5
        while not c._submit_after:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert c._submit_after > time.time()+50
        time.sleep(.05)
        assert server.attempts == 1 and not c._fatal
        with c._condition:
            c._submit_after = 0
            c._condition.notify_all()
        assert c.chat_json(messages)['content']['ok'] == 'rate-limited'
        assert len(server.posts) == 1
        assert server.posts[0]['requests'] == server.rejected['requests']
    finally:
        c.close()


def test_restart_does_not_replay_terminal_errors_forever(tmp_path, monkeypatch):
    server = Server()
    server.partial_failure = True
    messages = [[{'role': 'user', 'content': value}] for value in ('failed', 'kept')]
    first = client(tmp_path, monkeypatch, server)
    try:
        for message in messages:
            first.prefetch_json(message)
        with pytest.raises(RuntimeError):
            first.chat_json(messages[0])
        kept = first.chat_json(messages[1])
    finally:
        first.close()
    server.partial_failure = False
    resumed = client(tmp_path, monkeypatch, server)
    try:
        assert resumed.chat_json(messages[1])['batch']['id'] == kept['batch']['id']
        assert resumed.chat_json(messages[0])['content']['ok'] == 'failed'
        assert [len(post['requests']) for post in server.posts] == [2, 1]
    finally:
        resumed.close()


def test_failed_job_with_null_results_can_resume(tmp_path, monkeypatch):
    server = Server()
    messages = [{'role': 'user', 'content': 'retry failed job'}]
    write_json_atomic(tmp_path/'job_failed.json', {
        'journal': 'job_failed.json', 'status': 'failed',
        'request': {'requests': [{'custom_id': 'failed-request', 'body': {'messages': messages}}]},
        'remote': {'id': 'old-failed-job', 'status': 'failed', 'results': None},
    })
    resumed = client(tmp_path, monkeypatch, server)
    try:
        assert resumed.chat_json(messages)['content']['ok'] == 'retry failed job'
        assert len(server.posts) == 1
    finally:
        resumed.close()


def test_exhausted_format_replies_recover_then_allow_a_new_stage(tmp_path, monkeypatch):
    from predict.utils.response import call_with_json_validation, structured_response_is_valid
    class Malformed(Server):
        def get(self, url, *, timeout):
            response = super().get(url, timeout=timeout)
            if len(self.posts) <= 4:
                for row in response.payload['results']:
                    row['response']['body']['choices'][0]['message']['content'] = '{}'
            return response
    server = Malformed()
    messages = [{'role': 'user', 'content': 'bounded'}]
    first = client(tmp_path, monkeypatch, server)
    try:
        result = call_with_json_validation(first.chat_json, messages, required_fields=['ok'])
        assert not structured_response_is_valid(result) and len(server.posts) == 4
    finally:
        first.close()
    resumed = client(tmp_path, monkeypatch, server)
    try:
        result = call_with_json_validation(resumed.chat_json, messages, required_fields=['ok'])
        assert not structured_response_is_valid(result) and len(server.posts) == 4
        result = call_with_json_validation(resumed.chat_json, messages, required_fields=['ok'])
        assert structured_response_is_valid(result) and len(server.posts) == 5
    finally:
        resumed.close()
