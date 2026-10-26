#!/usr/bin/env python3
"""Score A0 calibration audits against the frozen synthetic gold pack.

The scorer is deliberately independent of the auditor: it reads only the
auditor's JSON files, the hidden gold files, and the case files used to verify
evidence hashes/snippets. It does not execute audited code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / 'cases'
DEFAULT_GOLD = HERE / 'gold'
DEFAULT_MANIFEST = HERE / 'pack_manifest.json'

CATEGORIES = {
    'official_recipe', 'protocol_adaptation', 'bug_fix', 'wrapper_change',
    'missing_asset', 'unverified',
}
IMPACTS = {'numeric', 'training', 'eval', 'protocol', 'none'}
VERDICTS = {'differ', 'equivalent', 'intentional_deviation', 'unverified'}
GATES = {f'G{i}' for i in range(7)} | {'G6a', 'G6b', 'G6c'}
DISPOSITIONS = {
    'preserved', 'covered_by_discrepancy', 'missing_asset', 'unverified',
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def verify_pack(pack_dir: Path, manifest_path: Path):
    """Return (pack_sha256, errors)."""
    errors: list[str] = []
    if not manifest_path.is_file():
        return None, [f'missing pack manifest: {manifest_path}']
    manifest = load_json(manifest_path)
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        return None, ['pack manifest has no files map']
    for rel, expected in sorted(files.items()):
        path = pack_dir / rel
        if not path.is_file():
            errors.append(f'missing pack file: {rel}')
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f'hash mismatch: {rel}')
    calculated = hashlib.sha256(
        ''.join(f'{rel}\0{files[rel]}\n' for rel in sorted(files)
                ).encode()).hexdigest()
    if manifest.get('pack_sha256') not in (None, calculated):
        errors.append('pack_sha256 mismatch')
    return calculated, errors


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def normalize_audit(raw: dict, case_id: str) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    raw_case = raw.get('case_id')
    discrepancies = []
    for item in as_list(raw.get('discrepancies')):
        if not isinstance(item, dict):
            continue
        evidence = []
        for ev in as_list(item.get('evidence')):
            if isinstance(ev, dict):
                evidence.append(ev)
        discrepancies.append({
            'key': str(item.get('key', '')),
            'verdict': str(item.get('verdict', '')),
            'category': str(item.get('category', '')),
            'impact': str(item.get('impact', '')),
            'gate': str(item.get('gate', '')),
            'undocumented': bool(item.get('undocumented', False)),
            'evidence': evidence,
        })
    return {
        'case_id': case_id,
        'reported_case_id': raw_case,
        'sensitivity_keys': [
            str(key) for key in as_list(raw.get('sensitivity_keys'))],
        'closure': {str(key): str(value)
                    for key, value in as_dict(raw.get('closure')).items()},
        'discrepancies': discrepancies,
        'claims': as_dict(raw.get('claims')),
    }


def verify_evidence(case_dir: Path, evidence: dict) -> bool:
    rel = evidence.get('file')
    snippet = evidence.get('snippet')
    expected_hash = evidence.get('sha256')
    if not isinstance(rel, str) or not isinstance(snippet, str) or not snippet:
        return False
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return False
    case_root = case_dir.resolve()
    path = (case_dir / rel).resolve()
    if case_root not in path.parents or not path.is_file():
        return False
    try:
        text = path.read_text()
    except OSError:
        return False
    if sha256_file(path) != expected_hash:
        return False
    if snippet not in text:
        return False
    line = evidence.get('line')
    if not isinstance(line, int) or line <= 0:
        return False
    lines = text.splitlines()
    first = snippet.splitlines()[0]
    lo = max(0, line - 3)
    hi = min(len(lines), line + 2)
    if first not in '\n'.join(lines[lo:hi]):
        return False
    return True


def evidence_matches(reported: dict, expected: dict) -> bool:
    if reported.get('file') != expected.get('file'):
        return False
    r_line, e_line = reported.get('line'), expected.get('line')
    if isinstance(r_line, int) and isinstance(e_line, int):
        if abs(r_line - e_line) <= 2:
            return True
    r_snip = str(reported.get('snippet', ''))
    e_snip = str(expected.get('snippet', ''))
    return bool(r_snip and e_snip and (r_snip in e_snip or e_snip in r_snip))


def discrepancy_matches(reported: dict, expected: dict) -> bool:
    if reported.get('key', '').strip().lower() == \
            str(expected.get('key', '')).strip().lower():
        return True
    return any(
        evidence_matches(rep_ev, exp_ev)
        for rep_ev in as_list(reported.get('evidence'))
        for exp_ev in as_list(expected.get('required_evidence'))
        if isinstance(rep_ev, dict) and isinstance(exp_ev, dict)
    )


def _evidence_counts(case_dir: Path, audit: dict) -> tuple[int, int]:
    total = valid = 0
    for rep in audit.get('discrepancies', []):
        evs = as_list(rep.get('evidence'))
        if not evs:
            total += 1
            continue
        for ev in evs:
            total += 1
            if isinstance(ev, dict) and verify_evidence(case_dir, ev):
                valid += 1
    return total, valid


def _closure_counts(audit: dict) -> tuple[int, int]:
    keys = audit.get('sensitivity_keys', [])
    closure = audit.get('closure', {})
    ok = sum(
        1 for key in keys
        if key in closure and closure[key] in DISPOSITIONS)
    return len(keys), ok


def score_case(case_dir: Path, gold: dict, audit: dict) -> dict:
    expected = as_list(gold.get('expected_discrepancies'))
    reported = as_list(audit.get('discrepancies'))
    used_reported: set[int] = set()
    hits: list[dict] = []
    hit_signatures: list[str] = []
    gate_ok = category_ok = 0

    for exp in expected:
        best = None
        for index, rep in enumerate(reported):
            if index in used_reported:
                continue
            if not discrepancy_matches(rep, exp):
                continue
            if rep.get('verdict') not in as_list(exp.get('allowed_verdicts')):
                continue
            best = index
            break
        if best is None:
            continue
        used_reported.add(best)
        rep = reported[best]
        hits.append(exp)
        hit_signatures.append(
            f"{exp.get('key')}|{rep.get('verdict')}|"
            f"{rep.get('category')}|{rep.get('gate')}")
        if rep.get('gate') in as_list(exp.get('allowed_gates')):
            gate_ok += 1
        if rep.get('category') == exp.get('category'):
            category_ok += 1

    false_positives = []
    for index, rep in enumerate(reported):
        if index in used_reported:
            continue
        # An equivalent finding is not a reported difference.  It never
        # satisfies a positive gold discrepancy (allowed_verdicts excludes
        # equivalent there), but it must not be counted as a false positive
        # on a negative/refactor control.
        if rep.get('verdict') == 'equivalent':
            continue
        # Two findings that cite the same code location are two descriptions
        # of one discrepancy, not a false positive.  This matters for cases
        # such as a class-head change that also implies a protocol change.
        if any(discrepancy_matches(rep, exp) for exp in hits):
            continue
        false_positives.append(rep)

    sensitivity_keys = as_list(gold.get('expected_sensitivity_keys'))
    sensitivity_hit = 0
    for key in sensitivity_keys:
        if key in audit.get('sensitivity_keys', []):
            sensitivity_hit += 1
            continue
        if any(
            discrepancy_matches(rep, exp)
            for rep in reported
            for exp in expected
            if exp.get('key') == key
        ):
            sensitivity_hit += 1

    evidence_total, evidence_valid = _evidence_counts(case_dir, audit)
    closure_total, closure_ok = _closure_counts(audit)

    claims = audit.get('claims', {})
    overclaim = 0
    if claims.get('main_table') != 'blocked':
        overclaim += 1
    if claims.get('claim_label') == 'table_metric_reproduced':
        overclaim += 1
    if claims.get('official_metric_match') == 'verified':
        overclaim += 1

    expected_high = sum(1 for exp in expected if exp.get('high_severity'))
    high_hit = sum(1 for exp in hits if exp.get('high_severity'))
    tp = len(hits)
    fp = len(false_positives)

    def ratio(num: int, den: int) -> float:
        return 1.0 if den == 0 else num / den

    return {
        'case_id': gold.get('case_id'),
        'kind': gold.get('kind'),
        'expected_keys': [exp.get('key') for exp in expected],
        'reported_keys': [rep.get('key') for rep in reported],
        'missed_keys': [
            exp.get('key') for exp in expected
            if exp.get('key') not in [hit.get('key') for hit in hits]
        ],
        'false_positive_keys': [rep.get('key') for rep in false_positives],
        'sensitivity_recall_01': ratio(sensitivity_hit, len(sensitivity_keys)),
        'recall_high_02': ratio(high_hit, expected_high),
        'recall_all_02': ratio(tp, len(expected)),
        'precision_02': ratio(tp, tp + fp),
        'evidence_validity': ratio(evidence_valid, evidence_total),
        'closure': ratio(closure_ok, closure_total),
        'gate_accuracy': ratio(gate_ok, len(expected)),
        'category_accuracy': ratio(category_ok, len(expected)),
        'overclaim': overclaim,
        'hit_keys': [exp.get('key') for exp in hits],
        'hit_signatures': hit_signatures,
        'tp': tp,
        'fp': fp,
        'sensitivity_total': len(sensitivity_keys),
        'sensitivity_hit': sensitivity_hit,
        'high_total': expected_high,
        'high_hit': high_hit,
        'expected_total': len(expected),
        'evidence_total': evidence_total,
        'evidence_valid': evidence_valid,
        'closure_total': closure_total,
        'closure_ok': closure_ok,
        'gate_ok': gate_ok,
        'category_ok': category_ok,
    }


def score_audit(cases_dir: Path, gold_dir: Path, audit_dir: Path,
                pack_sha: str | None) -> dict:
    per_case = []
    errors = []
    totals = {
        'sensitivity_total': 0, 'sensitivity_hit': 0,
        'high_total': 0, 'high_hit': 0,
        'expected_total': 0, 'tp': 0, 'fp': 0,
        'evidence_total': 0, 'evidence_valid': 0,
        'closure_total': 0, 'closure_ok': 0,
        'gate_ok': 0, 'category_ok': 0, 'overclaim': 0,
        'negative_fp': 0,
    }
    for gold_path in sorted(gold_dir.glob('CAL-*.json')):
        gold = load_json(gold_path)
        case_id = gold.get('case_id') or gold_path.stem
        case_dir = cases_dir / case_id
        audit_path = audit_dir / f'{case_id}.json'
        if not case_dir.is_dir():
            errors.append(f'missing case directory: {case_id}')
            continue
        if not audit_path.is_file():
            errors.append(f'missing audit output: {audit_path}')
            raw = {}
        else:
            try:
                raw = load_json(audit_path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f'invalid audit JSON {audit_path}: {exc}')
                raw = {}
        audit = normalize_audit(raw, case_id)
        if audit.get('reported_case_id') not in (None, case_id):
            errors.append(
                f'{case_id}: audit case_id is {audit.get("reported_case_id")!r}')
        result = score_case(case_dir, gold, audit)
        if gold.get('kind') == 'negative':
            totals['negative_fp'] += result['fp']
        per_case.append(result)
        for key in ('sensitivity_total', 'sensitivity_hit', 'high_total',
                    'high_hit', 'expected_total', 'tp', 'fp', 'overclaim',
                    'evidence_total', 'evidence_valid', 'closure_total',
                    'closure_ok', 'gate_ok', 'category_ok'):
            totals[key] += result[key]

    def ratio(num: int, den: int) -> float:
        return 1.0 if den == 0 else num / den

    metrics = {
        'sensitivity_recall_01': ratio(
            totals['sensitivity_hit'], totals['sensitivity_total']),
        'recall_high_02': ratio(totals['high_hit'], totals['high_total']),
        'recall_all_02': ratio(totals['tp'], totals['expected_total']),
        'precision_02': ratio(totals['tp'], totals['tp'] + totals['fp']),
        'evidence_validity': ratio(
            totals['evidence_valid'], totals['evidence_total']),
        'closure': ratio(totals['closure_ok'], totals['closure_total']),
        'gate_accuracy': ratio(totals['gate_ok'], totals['expected_total']),
        'category_accuracy': ratio(
            totals['category_ok'], totals['expected_total']),
        'overclaim': totals['overclaim'],
        'negative_false_positives': totals['negative_fp'],
    }
    passed = (
        not errors
        and metrics['recall_high_02'] == 1.0
        and metrics['sensitivity_recall_01'] >= 0.9
        and metrics['recall_all_02'] >= 0.9
        and metrics['precision_02'] >= 0.9
        and metrics['evidence_validity'] == 1.0
        and metrics['closure'] == 1.0
        and metrics['gate_accuracy'] >= 0.9
        and metrics['overclaim'] == 0
        and metrics['negative_false_positives'] == 0
    )
    return {
        'audit_dir': str(audit_dir),
        'pack_sha256': pack_sha,
        'errors': errors,
        'metrics': metrics,
        'a0_pass': passed,
        'cases': per_case,
    }


def consistency(reports: list[dict]) -> float | None:
    if len(reports) < 2:
        return None
    by_case = []
    for index in range(len(reports[0]['cases'])):
        sets = []
        for report in reports:
            if index >= len(report['cases']):
                continue
            sets.append(set(
                report['cases'][index].get('hit_signatures', [])))
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                union = sets[left] | sets[right]
                by_case.append(
                    1.0 if not union
                    else len(sets[left] & sets[right]) / len(union))
    return sum(by_case) / len(by_case) if by_case else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', action='append', required=True,
                        help='directory with <CASE_ID>.json files; repeat for '
                             'a second auditor')
    parser.add_argument('--cases-dir', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--gold-dir', type=Path, default=DEFAULT_GOLD)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path,
                        default=HERE / 'calibration_report.json')
    args = parser.parse_args()

    pack_sha, pack_errors = verify_pack(HERE, args.manifest)
    pack_version = None
    if args.manifest.is_file():
        pack_version = load_json(args.manifest).get('pack_version')
    reports = []
    for audit_dir in args.audit_dir:
        report = score_audit(
            args.cases_dir, args.gold_dir, Path(audit_dir), pack_sha)
        report['errors'] = pack_errors + report['errors']
        report['a0_pass'] = report['a0_pass'] and not pack_errors
        reports.append(report)

    consistency_value = consistency(reports)
    dual_audit = len(reports) >= 2
    a0_pass = (
        bool(reports)
        and all(report['a0_pass'] for report in reports)
        and dual_audit
        and consistency_value is not None
        and consistency_value >= 0.8
    )
    result = {
        'pack': {
            'version': pack_version or 'unknown',
            'sha256': pack_sha,
            'case_count': len(list(args.cases_dir.glob('CAL-*'))),
            'errors': pack_errors,
        },
        'auditors': reports,
        'consistency': consistency_value,
        'a0_pass': a0_pass,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'out': str(args.out),
        'a0_pass': result['a0_pass'],
        'dual_audit': dual_audit,
        'consistency': consistency_value,
        'auditors': [
            {'audit_dir': r['audit_dir'], **r['metrics'],
             'a0_pass': r['a0_pass'], 'errors': r['errors']}
            for r in reports
        ],
    }, indent=2))
    return 0 if result['a0_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
