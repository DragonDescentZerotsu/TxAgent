import concurrent.futures
import copy
import json

import pytest
import requests

from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.common.openrouter_batch import OpenRouterBatchClient
from tools.chembl_tool.common.openai_provider_pool import ProviderPoolConfig, ProviderSpec


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


class Server:
    def __init__(self):
        self.headers, self.jobs, self.posts = {}, {}, []
        self.partial_failure = False
        self.uncertain = False

    def post(self, url, *, data, timeout):
        assert url.endswith('/api/beta/batches')
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
    monkeypatch.setattr('tools.chembl_tool.common.openrouter_batch.requests.Session', lambda: server)
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
    assert valid.public_dict()['providers'][0]['temperature'] is None


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


def test_runner_uses_batch_with_actual_generation_settings(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as rt
    server = Server()
    monkeypatch.setattr('tools.chembl_tool.common.openrouter_batch.requests.Session', lambda: server)
    spec = ProviderSpec(name='batch', base_url='https://openrouter.ai/api/v1', model='openai/test',
        api_key_env='TEST_BATCH_KEY', max_inflight=2, transport='openrouter_batch', temperature=None,
        reasoning_effort='medium', batch_options={'flush_seconds': .01, 'poll_seconds': .01})
    config = ProviderPoolConfig(providers=(spec,), max_failovers=0)
    args = SimpleNamespace(output_root=str(tmp_path), retry_race_width=1, transport_max_retries=0,
        timeout_s=60, max_tokens=1024, tool_service_url='http://localhost')
    assert rt._generation_settings(config) == {'temperature': None, 'thinking': 'provider_default', 'reasoning_effort': 'medium'}
    pool = rt._make_client(args, config)
    try:
        result = pool.chat_json([{'role': 'user', 'content': 'integrated'}])
        assert result['content']['ok'] == 'integrated' and result['batch']['id'] == 'batch-1'
        assert result['execution_provider']['requested_model'] == 'openai/test'
    finally:
        pool.close()
    args.retry_race_width = 6
    with pytest.raises(ValueError, match='retry-race-width 1'):
        rt._make_client(args, config)
