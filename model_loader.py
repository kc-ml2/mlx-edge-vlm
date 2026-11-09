"""Shared local model loading, memory release, and idle timeout."""
import gc
import math
import os
import threading
import time

from mlx_vlm import load
from mlx_vlm.utils import load_config


def configure(application):
    """Bind to the running app without importing it (including __main__)."""
    global MODEL_DIR, runtime, state, state_lock, update, inference_lock
    MODEL_DIR = application.MODEL_DIR
    runtime = application.runtime
    state = application.state
    state_lock = application.state_lock
    update = application.update
    inference_lock = application.inference_lock


IDLE_TIMEOUT = float(os.environ.get('MODEL_IDLE_TIMEOUT', '300'))
last_used = 0.0


def touch():
    global last_used
    last_used = time.monotonic()


def start_idle_timer():
    if not math.isfinite(IDLE_TIMEOUT) or IDLE_TIMEOUT <= 0:
        raise ValueError('MODEL_IDLE_TIMEOUT must be positive (seconds).')
    threading.Thread(target=reap_idle, daemon=True).start()


class ModelError(Exception):
    def __init__(self, status, message, code):
        super().__init__(message)
        self.status, self.code = status, code


def local_model(repo):
    path = MODEL_DIR / repo
    # Basic presence check; the loader validates the remaining snapshot files.
    if not (path / 'config.json').is_file() or not any(path.glob('*.safetensors')):
        raise ModelError(404, f'Model {repo} is not available locally in {path}. Download it explicitly using the app first.', 'model_not_found')
    return path


def unload():
    # Caller holds inference_lock. Release Python references before clearing MLX cache.
    runtime.update(model=None, processor=None, config=None, repo=None)
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def load_model(repo, path):
    """Load a local snapshot. Caller holds inference_lock."""
    global last_used
    unload()
    config = load_config(str(path))
    model, processor = load(str(path))
    runtime.update(model=model, processor=processor, config=config, repo=repo)
    last_used = time.monotonic()
    update(phase='ready', message='Model ready', model_id=repo, path=str(path))


def load_local(repo):
    path = local_model(repo)
    with state_lock:
        if state['phase'] in ('downloading', 'loading'):
            raise ModelError(503, 'A model is being prepared. Retry later.', 'model_busy')
        if runtime['repo'] == repo and runtime['model'] is not None:
            return
        state.update(phase='loading', message='Loading local model...', model_id=repo)
    try:
        load_model(repo, path)
    except Exception:
        unload()
        update(phase='error', message='Could not load local model', model_id=repo)
        raise


def reap_idle():
    while True:
        time.sleep(1)
        with inference_lock:
            if runtime['model'] is not None and time.monotonic() - last_used >= IDLE_TIMEOUT:
                unload()
                with state_lock:
                    if state['phase'] == 'ready':
                        state.update(phase='idle', message='Model unloaded; next request will reload it')


