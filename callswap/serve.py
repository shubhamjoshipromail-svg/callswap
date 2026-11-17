"""Drop-in HTTP endpoint: answer locally when confident, otherwise forward to the original LLM."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import __version__
from .calibrate import predict
from .models import load_model


def _error(status, message, kind='invalid_request_error'):
    """Error body in the OpenAI shape, so existing clients surface the message."""
    return JSONResponse({'error': {'message': message, 'type': kind}}, status_code=status)


def _last_user_text(body):
    """Text of the last user message; supports plain strings and lists of content parts."""
    users = [m for m in body.get('messages') or [] if isinstance(m, dict) and m.get('role') == 'user']
    if not users:
        return None
    content = users[-1].get('content')
    if isinstance(content, list):
        content = ' '.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
    return content.strip() if isinstance(content, str) and content.strip() else None


def create_app(out, teacher_url=None, teacher_model=None, shadow=False, transport=None, teacher_timeout=30.0):
    """Build the FastAPI app for a run directory.

    The teacher key is read from the TEACHER_API_KEY environment variable. If it is not set, the caller's
    own Authorization header is forwarded, so an app can keep using the key it already has.
    """
    out = Path(out)
    model = load_model(out)
    cal = json.loads((out / 'calibration_result.json').read_text())
    sealed_path = out / 'sealed_result.json'
    verdict = json.loads(sealed_path.read_text())['verdict'] if sealed_path.exists() else None
    if shadow and not teacher_url:
        raise ValueError('Shadow mode needs a teacher: set --teacher-url (or TEACHER_URL)')
    log_path = out / 'requests.log'
    counts, shadow_seen = {}, {'compared': 0, 'agreed': 0}
    if log_path.exists():  # carry counters across restarts
        for line in log_path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts[entry.get('route')] = counts.get(entry.get('route'), 0) + 1
            if entry.get('agreed') is not None:
                shadow_seen['compared'] += 1
                shadow_seen['agreed'] += bool(entry['agreed'])

    def classify(text):
        labels, conf = predict(model, [text], cal['temperature'])
        return str(labels[0]), float(conf[0])

    def record(route, confidence, start, agreed=None, status=200):
        """One line per request: never the text or the label."""
        counts[route] = counts.get(route, 0) + 1
        if agreed is not None:
            shadow_seen['compared'] += 1
            shadow_seen['agreed'] += bool(agreed)
        entry = {'timestamp': datetime.now(timezone.utc).isoformat(), 'route': route, 'confidence': confidence,
                 'latency_ms': (time.monotonic() - start) * 1000, 'agreed': agreed, 'status': status}
        with log_path.open('a') as f:
            f.write(json.dumps(entry) + '\n')

    app = FastAPI(title='callswap', version=__version__)

    @app.get('/')
    def info():
        return {'name': 'callswap', 'version': __version__, 'labels': len(model.classes_), 'sealed_verdict': verdict,
                'threshold': cal['threshold'], 'teacher_configured': bool(teacher_url), 'shadow': shadow,
                'endpoints': ['POST /v1/chat/completions', 'POST /classify', 'GET /stats', 'GET /healthz']}

    @app.get('/healthz')
    def health():
        return {'status': 'ok'}

    @app.get('/stats')
    def stats():
        total = sum(counts.values())
        return {'requests': total, 'counts_by_route': counts,
                'local_share': counts.get('local', 0) / total if total else None,
                'shadow_compared': shadow_seen['compared'],
                'shadow_agreement_rate': shadow_seen['agreed'] / shadow_seen['compared'] if shadow_seen['compared'] else None}

    @app.post('/classify')
    async def classify_endpoint(request: Request):
        """Local prediction only; never calls the teacher."""
        start = time.monotonic()
        try:
            body = await request.json()
        except Exception:
            return _error(400, 'Request body must be JSON')
        text = body.get('text') if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return _error(400, 'A non-empty "text" string is required')
        label, confidence = await run_in_threadpool(classify, text)
        record('classify', confidence, start)
        return {'label': label, 'confidence': confidence, 'accepted': confidence >= cal['threshold'],
                'threshold': cal['threshold']}

    @app.post('/v1/chat/completions')
    async def completions(request: Request):
        start = time.monotonic()
        try:
            body = await request.json()
        except Exception:
            return _error(400, 'Request body must be JSON')
        if not isinstance(body, dict):
            return _error(400, 'Request body must be a JSON object')
        if body.get('stream'):
            return _error(400, 'callswap does not support streaming; send "stream": false')
        text = _last_user_text(body)
        if text is None:
            return _error(400, 'A user message with text content is required')
        label, confidence = await run_in_threadpool(classify, text)
        confident = confidence >= cal['threshold']
        headers = {'x-callswap-confidence': f'{confidence:.6f}'}

        if teacher_url and (shadow or not confident):
            payload = dict(body)
            if teacher_model:
                payload['model'] = teacher_model
            key = os.environ.get('TEACHER_API_KEY')
            auth = f'Bearer {key}' if key else request.headers.get('authorization')
            try:
                async with httpx.AsyncClient(transport=transport, timeout=teacher_timeout) as client:
                    upstream = await client.post(teacher_url, json=payload, headers={'Authorization': auth} if auth else {})
            except httpx.HTTPError as exc:
                record('teacher-error', confidence, start, status=502)
                return JSONResponse({'error': {'message': f'Teacher request failed: {type(exc).__name__}', 'type': 'teacher_error'}},
                                    status_code=502, headers={**headers, 'x-callswap-route': 'teacher-error'})
            agreed = None
            if shadow and upstream.status_code == 200:
                try:
                    agreed = upstream.json()['choices'][0]['message']['content'].strip() == label
                except Exception:
                    agreed = None  # teacher reply was not in the expected shape; nothing to compare
            route = 'teacher' if upstream.status_code == 200 else 'teacher-error'
            record(route, confidence, start, agreed, upstream.status_code)
            # Pass the teacher's reply through unchanged, including its error status if it failed.
            return Response(upstream.content, status_code=upstream.status_code,
                            media_type=upstream.headers.get('content-type', 'application/json'),
                            headers={**headers, 'x-callswap-route': route})

        # Answered locally. `local-unsure` means confidence was below the threshold but no teacher is configured.
        route = 'local' if confident else 'local-unsure'
        record(route, confidence, start)
        result = {'id': f'callswap-{time.time_ns()}', 'object': 'chat.completion', 'created': int(time.time()),
                  'model': 'callswap-local',
                  'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': label}, 'finish_reason': 'stop'}],
                  'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
        return JSONResponse(result, headers={**headers, 'x-callswap-route': route})

    return app
