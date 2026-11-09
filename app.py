#!/usr/bin/env python3
"""Local-only HTTP service for the captioning workspace."""
import base64
import binascii
import json
import io
import sys

import api
import model_loader
import os
import platform
import re
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / '.models'
DEFAULT_MODEL = 'mlx-community/Qwen2-VL-2B-Instruct-4bit'
MAX_BODY = 16 * 1024 * 1024
state_lock = threading.Lock()
inference_lock = threading.Lock()
state = {'phase': 'idle', 'message': 'No model loaded', 'model_id': None, 'path': None}
runtime = {'model': None, 'processor': None, 'config': None, 'repo': None}


def model_id(value):
    if not isinstance(value, str):
        raise ValueError('Model must be an owner/model ID.')
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
        MODEL_DIR.mkdir(exist_ok=True)
        path = snapshot_download(repo_id=repo, local_dir=str(MODEL_DIR / repo))
        update(phase='loading', message='Download complete. Loading model into memory...', path=path)
        with inference_lock:
            model_loader.load_model(repo, path)
    except Exception as exc:
        traceback.print_exc()
        update(phase='error', message=str(exc) or type(exc).__name__, model_id=repo, path=None)


def caption(image_data, prompt, max_tokens, repo=None):
    if not isinstance(image_data, str) or not image_data.startswith('data:image/') or ';base64,' not in image_data:
        raise ValueError('Expected a base64 image.')
    try:
        raw = base64.b64decode(image_data.split(',', 1)[1], validate=True)
    except (ValueError, binascii.Error):
        raise ValueError('Invalid image data.')
    if not raw or len(raw) > 10 * 1024 * 1024:
        raise ValueError('Image must be smaller than 10 MB.')
    with inference_lock:
        model_loader.load_local(repo or state['model_id'] or DEFAULT_MODEL)
        with tempfile.TemporaryDirectory(prefix='vlm-caption-') as directory:
            image_path = Path(directory) / 'frame.jpg'
            with Image.open(io.BytesIO(raw)) as img:
                img.convert('RGB').save(image_path, 'JPEG', quality=90)
            formatted = apply_chat_template(runtime['processor'], runtime['config'], prompt, num_images=1)
            try:
                result = generate(runtime['model'], runtime['processor'], formatted, [str(image_path)], max_tokens=max_tokens, verbose=False)
                return str(getattr(result, 'text', result)).strip()
            finally:
                model_loader.touch()


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
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object.')
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
                return self.send_json(200, {'caption': caption(data.get('image', ''), prompt, tokens)})
            return self.send_error(404)
        except model_loader.ModelError as exc:
            self.send_json(exc.status, {'error': str(exc)})
        except (ValueError, TypeError) as exc:
            self.send_json(400, {'error': str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self.send_json(500, {'error': str(exc) or type(exc).__name__})


model_loader.configure(sys.modules[__name__])
api.configure(sys.modules[__name__])


class SharedHandler(api.HandlerMixin, Handler):
    pass


if __name__ == '__main__':
    model_loader.start_idle_timer()
    port = int(os.environ.get('PORT', '8765'))
    server = ThreadingHTTPServer(('127.0.0.1', port), SharedHandler)
    print('Caption Studio running at http://127.0.0.1:%s' % port, flush=True)
    server.serve_forever()
