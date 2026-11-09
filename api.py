"""OpenAI-compatible request parsing and HTTP routes."""
import json
import time
import traceback
import uuid

import model_loader

DEFAULT_PROMPT = 'Describe the visible scene clearly and concisely. Focus on the main subjects and actions.'


def configure(application):
    global app
    app = application


def parse_chat(data):
    value = data.get('model', app.DEFAULT_MODEL)
    if not isinstance(value, str):
        raise ValueError('Model must be an owner/model ID.')
    repo = app.model_id(value)
    if data.get('stream', False) is not False:
        raise ValueError('Only stream=false is supported.')
    supported = {'model', 'messages', 'max_tokens', 'max_completion_tokens', 'stream'}
    unknown = set(data) - supported
    if unknown:
        raise ValueError('Unsupported parameters: ' + ', '.join(sorted(unknown)))
    messages = data.get('messages')
    if not isinstance(messages, list) or len(messages) not in (1, 2):
        raise ValueError('Expected one user message and an optional preceding system message.')
    if len(messages) == 2 and (not isinstance(messages[0], dict) or messages[0].get('role') != 'system'):
        raise ValueError('The first message must have role=system.')
    if not isinstance(messages[-1], dict) or messages[-1].get('role') != 'user':
        raise ValueError('The last message must have role=user.')
    prompt, images = [], []
    if len(messages) == 2:
        system = messages[0].get('content')
        if not isinstance(system, str) or not system.strip():
            raise ValueError('System content must be a nonempty string.')
        prompt.append({'role': 'system', 'content': system})
    content = messages[-1].get('content')
    if not isinstance(content, list):
        raise ValueError('User content must contain text and one image_url data URL.')
    texts = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError('Invalid content part.')
        if part.get('type') == 'text' and isinstance(part.get('text'), str):
            texts.append(part['text'])
        elif part.get('type') == 'image_url' and isinstance(part.get('image_url'), dict):
            images.append(part['image_url'].get('url'))
        else:
            raise ValueError('Only text and image_url content parts are supported.')
    text = '\n'.join(texts).strip() or DEFAULT_PROMPT
    if len(images) != 1 or not isinstance(images[0], str):
        raise ValueError('Exactly one base64 image data URL is required.')
    prompt.append({'role': 'user', 'content': text})
    if sum(len(m['content']) for m in prompt) > 2000:
        raise ValueError('Prompt must be at most 2000 characters.')
    tokens = data.get('max_completion_tokens', data.get('max_tokens', 100))
    if type(tokens) is not int or not 1 <= tokens <= 512:
        raise ValueError('max_tokens must be an integer between 1 and 512.')
    return repo, images[0], prompt, tokens


class HandlerMixin:
    def do_GET(self):
        if not self.path.startswith('/v1/'):
            return super().do_GET()
        if self.path == '/v1/models':
            models = []
            for config in sorted(model_loader.MODEL_DIR.glob('*/*/config.json')):
                repo = config.parent.relative_to(model_loader.MODEL_DIR).as_posix()
                try:
                    model_loader.local_model(repo)
                except model_loader.ModelError:
                    continue
                models.append({'id': repo, 'object': 'model', 'created': 0, 'owned_by': 'local'})
            return self.send_json(200, {'object': 'list', 'data': models})
        self.send_failure(404, 'Unknown endpoint.', 'not_found')

    def do_POST(self):
        if not self.path.startswith('/v1/'):
            return super().do_POST()
        # Local browser requests only. This also rejects cross-origin form submissions.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send_failure(403, 'Origin not allowed.', 'forbidden')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > app.MAX_BODY:
                raise ValueError('Request is too large or empty.')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object.')
            if self.path == '/v1/chat/completions':
                repo, image, prompt, tokens = parse_chat(data)
                text = app.caption(image, prompt, tokens, repo=repo)
                return self.send_json(200, {
                    'id': 'chatcmpl-' + uuid.uuid4().hex,
                    'object': 'chat.completion', 'created': int(time.time()), 'model': repo,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                                 'finish_reason': 'stop'}],
                })
            return self.send_failure(404, 'Unknown endpoint.', 'not_found')
        except model_loader.ModelError as exc:
            self.send_failure(exc.status, str(exc), exc.code)
        except (ValueError, TypeError) as exc:
            self.send_failure(400, str(exc), 'invalid_request')
        except Exception as exc:
            traceback.print_exc()
            self.send_failure(500, str(exc) or type(exc).__name__, 'inference_error')

    def send_failure(self, status, message, code):
        self.send_json(status, {'error': {
            'message': message, 'code': code, 'param': None,
            'type': 'invalid_request_error' if status < 500 else 'server_error',
        }})
