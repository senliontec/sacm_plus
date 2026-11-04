"""Tests for reconstructable code provenance (commit + diff + untracked)."""

import json
import subprocess

import core.run_artifacts as ra


def _git(repo, *args):
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout


def _init_repo(tmp_path):
    _git(tmp_path, 'init')
    _git(tmp_path, 'config', 'user.email', 'test@example.com')
    _git(tmp_path, 'config', 'user.name', 'Test')
    src = tmp_path / 'src'
    src.mkdir()
    model = src / 'model.py'
    model.write_text('value = 1\n')
    _git(tmp_path, 'add', 'src/model.py')
    _git(tmp_path, 'commit', '-m', 'initial')
    return model


def _snapshot(tmp_path, name):
    prov = tmp_path / name
    git = ra.git_info(tmp_path)
    meta = ra.write_code_provenance(prov, tmp_path, git_data=git)
    return prov, meta


def test_fingerprint_tracks_staged_and_untracked_code(tmp_path):
    model = _init_repo(tmp_path)

    prov1, meta1 = _snapshot(tmp_path, 'prov1')
    assert meta1['code_patch'] is None
    manifest1 = json.loads((prov1 / 'source_manifest.json').read_text())
    assert 'src/model.py' in manifest1

    # Staged change: plain `git diff` would miss it; `git diff HEAD` must not.
    model.write_text('value = 2\n')
    _git(tmp_path, 'add', 'src/model.py')
    prov2, meta2 = _snapshot(tmp_path, 'prov2')
    assert meta2['code_fingerprint'] != meta1['code_fingerprint']
    assert meta2['code_patch'] is not None
    assert '+value = 2' in (prov2 / 'code.patch').read_text()

    # Untracked source file: not in git diff; must still change fingerprint.
    new = tmp_path / 'src' / 'new_module.py'
    new.write_text('new_value = 1\n')
    prov3, meta3 = _snapshot(tmp_path, 'prov3')
    assert meta3['code_fingerprint'] != meta2['code_fingerprint']
    untracked = json.loads((prov3 / 'untracked.json').read_text())
    assert 'src/new_module.py' in untracked
    assert (prov3 / 'untracked_src' / 'src' / 'new_module.py').is_file()

    # Changing the untracked content must change the fingerprint again.
    new.write_text('new_value = 2\n')
    _, meta4 = _snapshot(tmp_path, 'prov4')
    assert meta4['code_fingerprint'] != meta3['code_fingerprint']


def test_clean_repo_has_commit_and_no_patch(tmp_path):
    _init_repo(tmp_path)
    prov, meta = _snapshot(tmp_path, 'prov')
    git = json.loads((ra.Path(prov) / 'code_version.json').read_text())
    assert git['commit']
    assert git['dirty'] is False
    assert meta['code_patch'] is None
    assert git['code_fingerprint'] == meta['code_fingerprint']
