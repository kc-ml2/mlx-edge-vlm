#!/usr/bin/env python3
"""Local-only HTTP service for the captioning workspace."""
import base64
import binascii
import json
import os
import platform
import re
import tempfile
import threading
import traceback
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / '.models'
DEFAULT_MODEL = 'mlx-community/Qwen2-VL-2B-Instruct-4bit'
MAX_BODY = 16 * 1024 * 1024
state_lock = threading.Lock()
inference_lock = threading.Lock()
state = {'phase': 'idle', 'message': 'No model loaded', 'model_id': None, 'path': None}
runtime = {'model': None, 'processor': None, 'config': None}


def model_id(value):
    value = value.strip()
    if value.startswith(('https://', 'http://')):
        url = urlparse(value)
        if url.hostname not in ('huggingface.co', 'www.huggingface.co') or url.query or url.fragment:
            raise ValueError('Enter a Hugging Face model URL or owner/model ID.')
        value = url.path.strip('/')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise ValueError('Use a model URL like https://huggingface.co/owner/model or owner/model.')
    if '..' in value:
        raise ValueError('Invalid model ID.')
    return value


def update(**values):
    with state_lock:
        state.update(values)


def prepare(repo):
    try:
        from huggingface_hub import snapshot_download
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        MODEL_DIR.mkdir(exist_ok=True)
        path = snapshot_download(repo_id=repo, local_dir=str(MODEL_DIR / repo))
        update(phase='loading', message='Download complete. Loading model into memory...', path=path)
        with inference_lock:
            runtime.update(model=None, processor=None, config=None)
            model, processor = load(path)
            config = load_config(path)
            runtime.update(model=model, processor=processor, config=config)
        update(phase='ready', message='Model ready', model_id=repo, path=path)
    except Exception as exc:
        traceback.print_exc()
        update(phase='error', message=str(exc) or type(exc).__name__, model_id=repo, path=None)


def decode_image(image_data):
    if not isinstance(image_data, str) or not image_data.startswith('data:image/') or ';base64,' not in image_data:
        raise ValueError('Expected a base64 image.')
    try:
        raw = base64.b64decode(image_data.split(',', 1)[1], validate=True)
    except (ValueError, binascii.Error):
        raise ValueError('Invalid image data.')
    if not raw or len(raw) > 10 * 1024 * 1024:
        raise ValueError('Image must be smaller than 10 MB.')
    return raw


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('Provide a conversation with at least one message.')
    if len(messages) > 40:
        raise ValueError('Conversation is too long.')
    normalized = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError('Each message must be an object with role and content.')
        role = str(item.get('role', '')).strip()
        content = str(item.get('content', '')).strip()
        if role not in ('user', 'assistant'):
            raise ValueError('Message role must be user or assistant.')
        if not content or len(content) > 2000:
            raise ValueError('Each message must be 1–2000 characters.')
        normalized.append({'role': role, 'content': content})
    if normalized[-1]['role'] != 'user':
        raise ValueError('The last message must be from the user.')
    return normalized


CANNOT_ANSWER = "I can't tell from this image. The details needed to answer that aren't clear enough in the scene."


def ask_turns(messages):
    """Split caption seed from later Q&A turns. Caption is not fed back into the ask prompt."""
    turns = []
    caption = ''
    saw_caption = False
    for message in messages:
        if not saw_caption:
            if message['role'] == 'assistant':
                caption = message['content']
                saw_caption = True
            continue
        turns.append(message)
    return caption, turns or [messages[-1]]


def build_ask_prompt(messages):
    """Build a short Q&A prompt. Omit the caption so the model cannot copy it."""
    _, turns = ask_turns(messages)
    question = turns[-1]['content']
    prior_questions = [m['content'] for m in turns[:-1] if m['role'] == 'user']
    parts = [
        'Look at the image and answer the question specifically.',
        'Give a short direct answer based on what is visible.',
        'Prefer answering when the image supports it. Do not write a full scene caption.',
        'Only if the image truly lacks the needed detail, say you cannot tell from the image and give one short reason.',
    ]
    if prior_questions:
        parts.append('Earlier questions:\n- ' + '\n- '.join(prior_questions))
    parts.append('Question: ' + question)
    parts.append('Direct answer:')
    return '\n\n'.join(parts)


def normalize_reply(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def is_repeat_reply(answer, banned):
    """True only for near-exact copies of the caption or an earlier reply."""
    norm = normalize_reply(answer)
    if not norm:
        return True
    for item in banned:
        other = normalize_reply(item)
        if not other:
            continue
        if norm == other:
            return True
        length_ratio = min(len(norm), len(other)) / max(len(norm), len(other))
        # Shared words are normal for real answers; only block near-duplicates.
        if length_ratio >= 0.8 and SequenceMatcher(None, norm, other).ratio() >= 0.9:
            return True
    return False


def finalize_ask_answer(answer, messages):
    if not answer:
        return CANNOT_ANSWER
    caption, turns = ask_turns(messages)
    banned = [caption] + [m['content'] for m in turns if m['role'] == 'assistant']
    if is_repeat_reply(answer, banned):
        return CANNOT_ANSWER
    return answer


def generate_reply(image_data, prompt, max_tokens):
    raw = decode_image(image_data)
    with inference_lock:
        if state['phase'] != 'ready' or runtime['model'] is None:
            raise ValueError('Load a model before generating captions.')
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image
        import io
        with tempfile.TemporaryDirectory(prefix='vlm-caption-') as directory:
            image_path = Path(directory) / 'frame.jpg'
            with Image.open(io.BytesIO(raw)) as img:
                img.convert('RGB').save(image_path, 'JPEG', quality=90)
            formatted = apply_chat_template(runtime['processor'], runtime['config'], prompt, num_images=1)
            result = generate(runtime['model'], runtime['processor'], formatted, [str(image_path)], max_tokens=max_tokens, verbose=False)
            return str(getattr(result, 'text', result)).strip()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, data):
        content = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == '/api/status':
            with state_lock:
                result = dict(state)
            result.update(architecture=platform.machine(), python=platform.python_version(), default_model=DEFAULT_MODEL)
            return self.send_json(200, result)
        files = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
        if self.path not in files:
            return self.send_error(404)
        name, content_type = files[self.path]
        content = (ROOT / 'static' / name).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        # Local browser requests only. This also rejects cross-origin form submissions.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send_json(403, {'error': 'Origin not allowed.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > MAX_BODY:
                raise ValueError('Request is too large or empty.')
            data = json.loads(self.rfile.read(length))
            if self.path == '/api/model':
                repo = model_id(data.get('model', ''))
                with state_lock:
                    if state['phase'] in ('downloading', 'loading'):
                        return self.send_json(409, {'error': 'A model is already being prepared.'})
                    state.update(phase='downloading', message='Downloading model from Hugging Face...', model_id=repo, path=None)
                threading.Thread(target=prepare, args=(repo,), daemon=True).start()
                return self.send_json(202, {'ok': True})
            if self.path == '/api/caption':
                prompt = str(data.get('prompt', '')).strip()
                if not prompt or len(prompt) > 2000:
                    raise ValueError('Enter a caption prompt (up to 2000 characters).')
                tokens = max(16, min(512, int(data.get('max_tokens', 100))))
                return self.send_json(200, {'caption': generate_reply(data.get('image', ''), prompt, tokens)})
            if self.path == '/api/ask':
                messages = normalize_messages(data.get('messages'))
                tokens = max(16, min(512, int(data.get('max_tokens', 160))))
                prompt = build_ask_prompt(messages)
                answer = generate_reply(data.get('image', ''), prompt, tokens)
                return self.send_json(200, {'answer': finalize_ask_answer(answer, messages)})
            return self.send_error(404)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {'error': str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self.send_json(500, {'error': str(exc) or type(exc).__name__})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8766'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print('Caption Studio running at http://127.0.0.1:%s' % port, flush=True)
    server.serve_forever()
