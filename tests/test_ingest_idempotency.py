"""Idempotency + validation unit tests for ingest_images (DATA-01, Pitfall F).

All network / GCS / PIL surfaces are mocked so this runs with only the stdlib
(it does not require the pinned heavy deps that Plan 01-01 installs). The fake
``requests`` / ``PIL`` / ``google.auth`` modules stay installed in
``sys.modules`` for the whole duration of each test (the ingest code imports
them lazily at *call* time), via the ``faked_env`` context manager.
"""

import contextlib
import importlib
import os
import sys
import types
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _build_fake_requests():
    fake = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    fake.RequestException = RequestException
    fake._next_response = None
    fake.get = lambda url, timeout=None, stream=False: fake._next_response
    return fake


def _make_response(status_code=200, content_type="image/jpeg", body=b"\xff\xd8\xff"):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
    }
    resp.iter_content = lambda chunk_size=1: iter([body])
    resp.content = body
    return resp


def _build_fake_pil(decodable=True):
    pil_pkg = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")

    class _Img:
        size = (100, 100)

        def verify(self):
            if not decodable:
                raise ValueError("undecodable")

    image_mod.MAX_IMAGE_PIXELS = 178_956_970
    image_mod.open = lambda *a, **k: _Img()
    pil_pkg.Image = image_mod
    return {"PIL": pil_pkg, "PIL.Image": image_mod}


@contextlib.contextmanager
def faked_env(extra_modules):
    """Install fake modules in sys.modules, (re)import ingest_images fresh.

    Keeps the fakes installed for the whole ``with`` body so lazily-imported
    ``requests`` / ``PIL`` resolve to the stubs at call time.
    """
    with mock.patch.dict(sys.modules, extra_modules):
        sys.modules.pop("mapclass.ingest_images", None)
        import mapclass.ingest_images as ing

        importlib.reload(ing)
        try:
            yield ing
        finally:
            sys.modules.pop("mapclass.ingest_images", None)


def _bucket_with_blob(exists, size):
    blob = mock.Mock()
    blob.exists.return_value = exists
    blob.size = size
    blob.reload = mock.Mock()
    blob.upload_from_string = mock.Mock()
    bucket = mock.Mock()
    bucket.blob.return_value = blob
    return bucket, blob


def test_existing_nonzero_blob_is_skipped_with_zero_uploads():
    mods = {"requests": _build_fake_requests(), **_build_fake_pil()}
    with faked_env(mods) as ing:
        bucket, blob = _bucket_with_blob(exists=True, size=12345)
        status = ing.mirror_one(
            bucket, {"id": "RUMSEY~X", "image_url": "http://x"}
        )
        assert status == ing.STATUS_SKIPPED
        blob.upload_from_string.assert_not_called()


def test_non_200_yields_dead_url_and_no_upload():
    fake = _build_fake_requests()
    fake._next_response = _make_response(status_code=404)
    mods = {"requests": fake, **_build_fake_pil()}
    with faked_env(mods) as ing:
        bucket, blob = _bucket_with_blob(exists=False, size=None)
        with mock.patch.object(ing.time, "sleep", lambda *_: None):
            status = ing.mirror_one(
                bucket, {"id": "RUMSEY~Y", "image_url": "http://dead"}
            )
        assert status == ing.STATUS_DEAD_URL
        blob.upload_from_string.assert_not_called()


def test_undecodable_bytes_yield_decode_fail_and_no_upload():
    fake = _build_fake_requests()
    fake._next_response = _make_response(status_code=200, body=b"not-an-image")
    mods = {"requests": fake, **_build_fake_pil(decodable=False)}
    with faked_env(mods) as ing:
        bucket, blob = _bucket_with_blob(exists=False, size=None)
        status = ing.mirror_one(
            bucket, {"id": "RUMSEY~Z", "image_url": "http://bad"}
        )
        assert status == ing.STATUS_DECODE_FAIL
        blob.upload_from_string.assert_not_called()


def test_bad_content_type_yields_status_and_no_upload():
    fake = _build_fake_requests()
    fake._next_response = _make_response(
        status_code=200, content_type="text/html"
    )
    mods = {"requests": fake, **_build_fake_pil()}
    with faked_env(mods) as ing:
        bucket, blob = _bucket_with_blob(exists=False, size=None)
        status = ing.mirror_one(
            bucket, {"id": "RUMSEY~H", "image_url": "http://html"}
        )
        assert status == ing.STATUS_BAD_CONTENT_TYPE
        blob.upload_from_string.assert_not_called()


def test_ok_path_uploads_once_and_returns_ok():
    fake = _build_fake_requests()
    fake._next_response = _make_response(status_code=200)
    mods = {"requests": fake, **_build_fake_pil()}
    with faked_env(mods) as ing:
        bucket, blob = _bucket_with_blob(exists=False, size=None)
        status = ing.mirror_one(
            bucket, {"id": "RUMSEY~OK", "image_url": "http://good"}
        )
        assert status == ing.STATUS_OK
        assert blob.upload_from_string.call_count == 1


def test_verify_adc_raises_clear_error_when_credentials_absent():
    google_pkg = types.ModuleType("google")
    auth_mod = types.ModuleType("google.auth")
    exc_mod = types.ModuleType("google.auth.exceptions")

    class DefaultCredentialsError(Exception):
        pass

    exc_mod.DefaultCredentialsError = DefaultCredentialsError

    def _default():
        raise DefaultCredentialsError("no ADC")

    auth_mod.default = _default
    auth_mod.exceptions = exc_mod
    google_pkg.auth = auth_mod

    mods = {
        "google": google_pkg,
        "google.auth": auth_mod,
        "google.auth.exceptions": exc_mod,
        "requests": _build_fake_requests(),
        **_build_fake_pil(),
    }
    with faked_env(mods) as ing:
        try:
            ing.verify_adc()
            assert False, "expected verify_adc to raise"
        except RuntimeError as exc:
            assert "Application Default Credentials" in str(exc)
