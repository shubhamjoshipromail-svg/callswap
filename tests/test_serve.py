import json

import httpx
import pytest
from conftest import set_threshold
from fastapi.testclient import TestClient

from callswap.serve import create_app

TEXT = 'alpha distinctive topic sample number 7 SECRETWORD'
BODY = {'model': 'big-llm', 'messages': [{'role': 'system', 'content': 'Classify.'}, {'role': 'user', 'content': TEXT}]}
URL = 'https://teacher.test/v1/chat/completions'


def teacher(answer='alpha', status=200, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        if status != 200:
            return httpx.Response(status, json={'error': {'message': 'teacher is down'}})
        return httpx.Response(200, json={'id': 'up-1', 'choices': [{'message': {'role': 'assistant', 'content': answer}}]})
    return httpx.MockTransport(handler)


def post(client, body=BODY, **kwargs):
    return client.post('/v1/chat/completions', json=body, **kwargs)


def test_confident_request_is_answered_locally(run_dir):
    set_threshold(run_dir, 0.0)
    seen = []
    client = TestClient(create_app(run_dir, URL, transport=teacher(seen=seen)))
    response = post(client)
    data = response.json()
    assert response.status_code == 200 and response.headers['x-callswap-route'] == 'local'
    assert data['choices'][0]['message'] == {'role': 'assistant', 'content': 'alpha'}
    assert data['object'] == 'chat.completion' and data['choices'][0]['finish_reason'] == 'stop' and 'usage' in data
    assert 0 <= float(response.headers['x-callswap-confidence']) <= 1
    assert seen == []  # the teacher was not called


def test_unsure_request_falls_back_to_teacher_with_original_body(run_dir, monkeypatch):
    monkeypatch.delenv('TEACHER_API_KEY', raising=False)
    set_threshold(run_dir, 2.0)
    seen = []
    client = TestClient(create_app(run_dir, URL, transport=teacher('from-teacher', seen=seen)))
    response = post(client, headers={'Authorization': 'Bearer caller-key'})
    assert response.headers['x-callswap-route'] == 'teacher'
    assert response.json()['choices'][0]['message']['content'] == 'from-teacher' and response.json()['id'] == 'up-1'
    assert json.loads(seen[0].content) == BODY and str(seen[0].url) == URL
    assert seen[0].headers['authorization'] == 'Bearer caller-key'  # caller's own key is forwarded


def test_teacher_key_from_environment_and_model_override(run_dir, monkeypatch):
    monkeypatch.setenv('TEACHER_API_KEY', 'env-key')
    set_threshold(run_dir, 2.0)
    seen = []
    client = TestClient(create_app(run_dir, URL, teacher_model='gpt-x', transport=teacher(seen=seen)))
    post(client, headers={'Authorization': 'Bearer caller-key'})
    assert seen[0].headers['authorization'] == 'Bearer env-key'
    assert json.loads(seen[0].content)['model'] == 'gpt-x'


def test_no_key_anywhere_sends_no_authorization_header(run_dir, monkeypatch):
    monkeypatch.delenv('TEACHER_API_KEY', raising=False)
    set_threshold(run_dir, 2.0)
    seen = []
    post(TestClient(create_app(run_dir, URL, transport=teacher(seen=seen))))
    assert 'authorization' not in seen[0].headers


def test_unsure_without_teacher_is_marked_local_unsure(run_dir):
    set_threshold(run_dir, 2.0)
    response = post(TestClient(create_app(run_dir)))
    assert response.status_code == 200 and response.headers['x-callswap-route'] == 'local-unsure'
    assert response.json()['choices'][0]['message']['content'] in ('alpha', 'beta', 'gamma')


def test_shadow_always_uses_teacher_and_records_agreement(run_dir):
    set_threshold(run_dir, 0.0)  # local is confident, shadow must still ask the teacher
    client = TestClient(create_app(run_dir, URL, shadow=True, transport=teacher('alpha')))
    response = post(client)
    assert response.headers['x-callswap-route'] == 'teacher' and response.json()['id'] == 'up-1'
    assert client.get('/stats').json()['shadow_agreement_rate'] == 1
    disagreeing = TestClient(create_app(run_dir, URL, shadow=True, transport=teacher('beta')))
    post(disagreeing)
    stats = disagreeing.get('/stats').json()  # counters are restored from the log, then updated
    assert stats['shadow_compared'] == 2 and stats['shadow_agreement_rate'] == .5
    assert stats['counts_by_route'] == {'teacher': 2}


def test_shadow_without_teacher_is_refused(run_dir):
    with pytest.raises(ValueError, match='teacher'):
        create_app(run_dir, shadow=True)


def test_teacher_error_status_is_passed_through(run_dir):
    set_threshold(run_dir, 2.0)
    client = TestClient(create_app(run_dir, URL, transport=teacher(status=500)))
    response = post(client)
    assert response.status_code == 500 and response.headers['x-callswap-route'] == 'teacher-error'
    assert response.json()['error']['message'] == 'teacher is down'


def test_teacher_unreachable_gives_502(run_dir):
    set_threshold(run_dir, 2.0)

    def refuse(request):
        raise httpx.ConnectError('no route')
    client = TestClient(create_app(run_dir, URL, transport=httpx.MockTransport(refuse)))
    response = post(client)
    assert response.status_code == 502 and response.json()['error']['type'] == 'teacher_error'
    assert client.get('/stats').json()['counts_by_route'] == {'teacher-error': 1}


@pytest.mark.parametrize('body, fragment', [
    ({**BODY, 'stream': True}, 'streaming'),
    ({'model': 'm', 'messages': []}, 'user message'),
    ({'model': 'm', 'messages': [{'role': 'system', 'content': 'only system'}]}, 'user message'),
    ({'model': 'm', 'messages': [{'role': 'user', 'content': '   '}]}, 'user message'),
    ({'model': 'm'}, 'user message'),
    ([1, 2], 'JSON object'),
])
def test_bad_requests_get_400(run_dir, body, fragment):
    response = post(TestClient(create_app(run_dir)), body)
    assert response.status_code == 400 and fragment in response.json()['error']['message']


def test_invalid_json_gets_400(run_dir):
    client = TestClient(create_app(run_dir))
    response = client.post('/v1/chat/completions', content=b'{not json', headers={'content-type': 'application/json'})
    assert response.status_code == 400


def test_content_parts_and_last_user_message(run_dir):
    set_threshold(run_dir, 0.0)
    client = TestClient(create_app(run_dir))
    body = {'messages': [{'role': 'user', 'content': 'beta distinctive topic sample number 3'},
                         {'role': 'assistant', 'content': 'beta'},
                         {'role': 'user', 'content': [{'type': 'text', 'text': 'gamma distinctive topic sample number 5'}]}]}
    assert post(client, body).json()['choices'][0]['message']['content'] == 'gamma'


def test_classify_endpoint_and_info(run_dir):
    client = TestClient(create_app(run_dir))
    result = client.post('/classify', json={'text': 'gamma distinctive topic sample number 5'}).json()
    assert result['label'] == 'gamma' and isinstance(result['accepted'], bool) and 0 <= result['confidence'] <= 1
    assert client.post('/classify', json={'text': ''}).status_code == 400
    assert client.post('/classify', json={'nope': 1}).status_code == 400
    assert client.get('/healthz').json() == {'status': 'ok'}
    info = client.get('/').json()
    assert info['labels'] == 3 and info['teacher_configured'] is False and info['sealed_verdict'] in ('PASS', 'FAIL')


def test_log_has_no_text_or_label_and_stats_persist(run_dir):
    set_threshold(run_dir, 0.0)
    client = TestClient(create_app(run_dir))
    post(client)
    post(client)
    log = (run_dir / 'requests.log').read_text()
    assert 'SECRETWORD' not in log and 'distinctive' not in log and 'alpha' not in log
    entries = [json.loads(line) for line in log.splitlines()]
    assert [e['route'] for e in entries] == ['local', 'local']
    assert set(entries[0]) == {'timestamp', 'route', 'confidence', 'latency_ms', 'agreed', 'status'}
    restarted = TestClient(create_app(run_dir))
    stats = restarted.get('/stats').json()
    assert stats['requests'] == 2 and stats['local_share'] == 1 and stats['shadow_agreement_rate'] is None
