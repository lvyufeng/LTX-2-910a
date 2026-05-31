from pathlib import Path

from ltx_core import accelerator


def test_tbe_pythonpath_status_warns_when_cann_paths_are_missing(monkeypatch, tmp_path):
    cann_root = tmp_path / "cann-9.0.0"
    python_site = cann_root / "python" / "site-packages"
    tbe_impl = cann_root / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"
    python_site.mkdir(parents=True)
    tbe_impl.mkdir(parents=True)
    project_src = tmp_path / "repo" / "packages" / "ltx-core" / "src"
    project_src.mkdir(parents=True)

    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_root))
    monkeypatch.setenv("PYTHONPATH", str(project_src))

    status, detail = accelerator._tbe_pythonpath_status()

    assert status == "WARN"
    assert str(python_site) in detail
    assert str(tbe_impl) in detail


def test_tbe_pythonpath_status_ok_when_set_env_entries_are_preserved(monkeypatch, tmp_path):
    cann_root = tmp_path / "cann-9.0.0"
    python_site = cann_root / "python" / "site-packages"
    tbe_impl = cann_root / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"
    python_site.mkdir(parents=True)
    tbe_impl.mkdir(parents=True)
    project_src = tmp_path / "repo" / "packages" / "ltx-core" / "src"
    project_src.mkdir(parents=True)

    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_root))
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{project_src}:{python_site}:{tbe_impl}",
    )

    status, detail = accelerator._tbe_pythonpath_status()

    assert status == "OK"
    assert detail is None


def test_pythonpath_has_normalizes_paths(monkeypatch, tmp_path):
    (tmp_path / "a").mkdir()
    path = tmp_path / "a" / ".." / "cann-python"
    path.resolve().mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", str(path.resolve()))

    assert accelerator._pythonpath_has(str(path))
