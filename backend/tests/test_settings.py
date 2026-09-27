import pytest

from app import settings


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / "backend.env"
    path.write_text("GEMINI_API_KEY=test-file-key\nGEMINI_MODEL=gemini-3.6-flash\n", encoding="utf-8-sig")
    monkeypatch.setattr(settings, "ENV_FILE", path)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.delenv("GEMINI_MODEL")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GEMINI_API_KEY")
    settings.get_settings.cache_clear()
    yield path
    settings.get_settings.cache_clear()


def test_loads_backend_env_from_another_working_directory(env_file, tmp_path, monkeypatch):
    import os

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    settings.get_settings()
    assert os.environ["GEMINI_API_KEY"] == "test-file-key"
    assert os.environ["GEMINI_MODEL"] == "gemini-3.6-flash"


def test_explicit_environment_has_precedence(env_file, monkeypatch):
    import os

    monkeypatch.setenv("GEMINI_API_KEY", "test-process-key")
    settings.get_settings()
    assert os.environ["GEMINI_API_KEY"] == "test-process-key"
