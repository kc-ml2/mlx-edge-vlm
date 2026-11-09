import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as application
import api
import model_loader as loader


class APITests(unittest.TestCase):
    def request(self):
        return {'model': application.DEFAULT_MODEL, 'messages': [
            {'role': 'system', 'content': 'Caption accurately.'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Describe this.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,YQ=='}}]}]}

    def test_import_order(self):
        import subprocess
        import sys
        for imports in ('import app; import api', 'import api; import app'):
            subprocess.run([sys.executable, '-c', imports + '; import model_loader; assert model_loader.runtime is app.runtime'], check=True)

    def test_busy_model_returns_503(self):
        with patch.object(loader, 'local_model', return_value=Path('/unused')), patch.dict(
            loader.state, phase='downloading'
        ):
            with self.assertRaises(loader.ModelError) as caught:
                loader.load_local(application.DEFAULT_MODEL)
            self.assertEqual(caught.exception.status, 503)

    def test_shared_runtime_and_lock(self):
        self.assertIs(loader.runtime, application.runtime)
        self.assertIs(loader.inference_lock, application.inference_lock)

    def test_ui_routes_delegate(self):
        handler = object.__new__(application.SharedHandler)
        handler.path = '/api/status'
        with patch.object(application.Handler, 'do_GET') as get:
            handler.do_GET()
            get.assert_called_once()
        handler.path = '/api/caption'
        with patch.object(application.Handler, 'do_POST') as post:
            handler.do_POST()
            post.assert_called_once()

    def test_recent_use_prevents_unload(self):
        with patch.object(loader.time, 'sleep', side_effect=[None, InterruptedError]), patch.object(
            loader, 'last_used', 999
        ), patch.object(loader.time, 'monotonic', return_value=1000), patch.dict(
            loader.runtime, model=object()
        ), patch.object(loader, 'unload') as unload:
            with self.assertRaises(InterruptedError):
                loader.reap_idle()
            unload.assert_not_called()

    def test_chat_preserves_roles(self):
        repo, image, prompt, tokens = api.parse_chat(self.request())
        self.assertEqual(repo, application.DEFAULT_MODEL)
        self.assertEqual([m['role'] for m in prompt], ['system', 'user'])
        self.assertEqual(tokens, 100)

    def test_rejects_unsupported(self):
        for extra in ({'stream': True}, {'tools': []}, {'max_tokens': 0}, {'model': None}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                api.parse_chat({**self.request(), **extra})

    def test_missing_model_does_not_load(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(loader, 'MODEL_DIR', Path(directory)):
            with self.assertRaises(loader.ModelError) as caught:
                loader.load_local(application.DEFAULT_MODEL)
            self.assertEqual(caught.exception.status, 404)
            self.assertEqual(caught.exception.code, 'model_not_found')

    def test_warm_model_reused(self):
        with patch.object(loader, 'local_model', return_value=Path('/unused')), patch.dict(
            loader.runtime, model=object(), repo=application.DEFAULT_MODEL
        ), patch.object(loader, 'unload') as unload:
            loader.load_local(application.DEFAULT_MODEL)
            unload.assert_not_called()

    def test_idle_unload(self):
        # Stop the infinite worker on its second iteration.
        with patch.object(loader.time, 'sleep', side_effect=[None, InterruptedError]), patch.object(
            loader, 'last_used', 0
        ), patch.object(loader.time, 'monotonic', return_value=1000), patch.object(
            loader, 'IDLE_TIMEOUT', 300
        ), patch.dict(loader.runtime, model=object()), patch.dict(loader.state, phase='ready'), patch.object(
            loader, 'unload'
        ) as unload:
            with self.assertRaises(InterruptedError):
                loader.reap_idle()
            unload.assert_called_once()
            self.assertEqual(loader.state['phase'], 'idle')


if __name__ == '__main__':
    unittest.main()
