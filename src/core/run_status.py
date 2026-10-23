"""Lightweight, crash-safe run status for long CLI jobs.

A :class:`RunStatus` atomically writes a small JSON file (by default from
``SACM_RUN_STATUS``) and refreshes it from a daemon heartbeat thread.  The
goal is simple: at any moment a user can answer "which stage is it in, how
far has it got, and how long is left?" without inspecting PIDs or staring at
a silent log.

Status writes are strictly best-effort: a permissions/disk error never stops
training or evaluation.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(
        timespec='seconds')


class RunStatus:
    """Write a heartbeat + progress JSON file for one running pipeline.

    Parameters
    ----------
    path:
        Status file.  Falls back to ``$SACM_RUN_STATUS``.  If neither is set,
        the object is a no-op (progress printing still works).
    heartbeat:
        Seconds between heartbeat writes.  Use ``0`` for no background thread
        (for example in a parent process that only writes stage transitions).
    """

    def __init__(self, path=None, method=None, stage=None, stage_index=None,
                 stage_total=None, heartbeat=10.0):
        env_path = os.environ.get('SACM_RUN_STATUS')
        chosen = path or env_path
        self.path = Path(chosen) if chosen else None
        previous = {}
        if self.path is not None and self.path.is_file():
            try:
                previous = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                previous = {}
        if stage_index is None:
            stage_index = previous.get('stage_index')
        if stage_total is None:
            stage_total = previous.get('stage_total')
        now = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._state = {
            'method': method or os.environ.get('SACM_RUN_METHOD') or 'unknown',
            'status': 'running',
            'stage': stage,
            'stage_index': stage_index,
            'stage_total': stage_total,
            'dataset': None,
            'dataset_index': None,
            'datasets_total': None,
            'datasets_done': 0,
            'dataset_done': 0,
            'dataset_total': 0,
            'overall_done': 0,
            'overall_total': 0,
            'active_datasets': [],
            'rate': 0.0,
            'eta_s': None,
            'started_epoch': now,
            'stage_started_epoch': now,
            'dataset_started_epoch': now,
            'updated_epoch': now,
            'started_at': iso_utc(now),
            'updated_at': iso_utc(now),
            'elapsed_s': 0.0,
            'stage_elapsed_s': 0.0,
            'pid': os.getpid(),
            'message': '',
            'error': None,
        }
        with self._lock:
            self._write_locked()
        if self.path is not None and heartbeat and heartbeat > 0:
            self._thread = threading.Thread(
                target=self._heartbeat, args=(float(heartbeat),),
                name='run-status-heartbeat', daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ API
    def refresh(self):
        """Reload the on-disk state (used by a parent after a child stage)."""
        if self.path is None:
            return self
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return self
        with self._lock:
            self._state.update(data)
        return self

    def update(self, **fields):
        """Merge fields and write immediately."""
        with self._lock:
            self._state.update(fields)
            self._write_locked()
        return self

    def set_stage(self, stage, stage_index=None, stage_total=None, **fields):
        with self._lock:
            now = time.time()
            self._state.update({
                'stage': stage,
                'stage_index': stage_index,
                'stage_total': stage_total,
                'stage_started_epoch': now,
                'stage_elapsed_s': 0.0,
                'dataset': None,
                'dataset_index': None,
                'datasets_done': 0,
                'dataset_done': 0,
                'dataset_total': 0,
                'overall_done': 0,
                'active_datasets': [],
                'rate': 0.0,
                'eta_s': None,
                'started_epoch': now,
                'started_at': iso_utc(now),
                'status': 'running',
                'error': None,
                'message': '',
            })
            self._state.update(fields)
            self._write_locked()
        return self

    def start_dataset(self, dataset, index=None, total=0,
                      overall_total=None, **fields):
        with self._lock:
            now = time.time()
            self._state.update({
                'dataset': dataset,
                'dataset_index': index,
                'dataset_done': 0,
                'dataset_total': total,
                'dataset_started_epoch': now,
                'active_datasets': [dataset],
                'rate': 0.0,
                'eta_s': None,
            })
            if overall_total is not None:
                self._state['overall_total'] = overall_total
            self._state.update(fields)
            self._write_locked()
        return self

    def advance(self, n=1, overall=True, write=True):
        with self._lock:
            s = self._state
            s['dataset_done'] += n
            if overall:
                s['overall_done'] += n
            elapsed = max(time.time() - s['dataset_started_epoch'], 1e-6)
            rate = s['dataset_done'] / elapsed if s['dataset_done'] else 0.0
            s['rate'] = round(rate, 2)
            if rate > 0 and s['dataset_total']:
                remaining = max(s['dataset_total'] - s['dataset_done'], 0)
                s['eta_s'] = round(remaining / rate, 1)
            if write:
                self._write_locked()
        return self

    def finish_dataset(self, **fields):
        with self._lock:
            s = self._state
            s['datasets_done'] += 1
            if s['dataset_total']:
                s['dataset_done'] = s['dataset_total']
            s['active_datasets'] = []
            s['eta_s'] = 0.0
            s['message'] = f"{s['dataset']} complete"
            s.update(fields)
            self._write_locked()
        return self

    def set_active(self, names, **fields):
        return self.update(active_datasets=list(names), **fields)

    def finish(self, message='done'):
        with self._lock:
            now = time.time()
            self._state.update({
                'status': 'done',
                'message': message,
                'active_datasets': [],
                'eta_s': 0.0,
                'finished_epoch': now,
                'finished_at': iso_utc(now),
            })
            self._write_locked()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self

    def fail(self, error):
        with self._lock:
            now = time.time()
            self._state.update({
                'status': 'failed',
                'error': str(error),
                'message': 'failed',
                'active_datasets': [],
                'finished_epoch': now,
                'finished_at': iso_utc(now),
            })
            self._write_locked()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._write_locked()

    # -------------------------------------------------------------- internal
    def _heartbeat(self, interval):
        while not self._stop.wait(interval):
            with self._lock:
                self._write_locked()

    def _write_locked(self):
        if self.path is None:
            return
        now = time.time()
        s = self._state
        s['updated_epoch'] = now
        s['updated_at'] = iso_utc(now)
        s['elapsed_s'] = round(now - s['started_epoch'], 1)
        s['stage_elapsed_s'] = round(now - s['stage_started_epoch'], 1)
        s['pid'] = os.getpid()
        tmp = self.path.with_name(self.path.name + '.tmp')
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, 'w') as f:
                json.dump(s, f, indent=2, ensure_ascii=False)
                f.write('\n')
            os.replace(tmp, self.path)
        except OSError:
            # Status is auxiliary; never let it break the actual job.
            pass
