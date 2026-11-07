from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from Experiment.SCOPE.Stage2.Task22.actor import SpatialQueryActor
from Experiment.SCOPE.Stage2.Task233.analyze import summarize_anchor, write_csv
from Experiment.SCOPE.Stage2.Task233.spatial_warmup import evaluate, load_samples


def verify(output):
    output = output.resolve()
    config = json.loads((output / 'RUN_CONFIG.json').read_text())
    audit = json.loads((output / 'OFFLINE_AUDIT.json').read_text())
    manifest = json.loads((output / 'TRAIN_MANIFEST.json').read_text())
    selection = json.loads((output / 'INITIALIZATION_SELECTION.json').read_text())
    dataset = Path(config['dataset'])
    samples = load_samples(dataset, 'cpu')
    anchors = json.loads((dataset / 'ANCHOR_MANIFEST_256.json').read_text())['anchors']
    quality = {row['anchor_id']: row for row in csv.DictReader((dataset / 'candidate_quality_256.csv').open())}
    groups = defaultdict(list)
    for path in {anchor['evidence'] for anchor in anchors}:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            groups[row['anchor_id']].append(row)
    thresholds = json.loads((ROOT / 'artifacts/gate0_v3/FROZEN_THRESHOLDS.json').read_text())
    label_checks = []
    for anchor in anchors:
        expected = summarize_anchor(groups[anchor['anchor_id']], thresholds)
        stored = quality[anchor['anchor_id']]
        label_checks.append(expected['teacher_variant'] == stored['teacher_variant'] and
                            expected['confident'] == (stored['confident'] == 'True') and
                            abs(expected['gt_margin'] - float(stored['gt_margin'])) < 1e-12)
    inputs = json.loads((output / 'INPUT_HASHES.json').read_text())
    substitutions = {}
    changed = []
    for path, expected in inputs.items():
        actual = sha256(path)
        if actual != expected:
            snapshot = output / 'source_snapshot' / Path(path).name
            if Path(path) == ROOT / 'spatial_warmup.py' and snapshot.is_file() and sha256(snapshot) == expected:
                substitutions[path] = {'executed_source_snapshot': str(snapshot), 'executed_sha256': expected,
                                       'current_sha256': actual, 'reason': 'post-run guard prevents modifying existing outputs on rejected rerun'}
            else:
                changed.append(path)
    checkpoints = {}
    stop_checks = {}
    optimizer_counts = {}
    for phase in ('task22_reset', 'update14_reset', 'warmup'):
        payload = torch.load(output / f'{phase}_last.pt', map_location='cpu', weights_only=False)
        checkpoints[phase] = payload
        initial_label = selection['chosen'] if phase == 'warmup' else phase
        initial = torch.load(output / f'{initial_label}_initial.pt', map_location='cpu', weights_only=True)['actor']
        stop_checks[phase] = all(torch.equal(value, initial[name]) for name, value in payload['actor'].items() if name.startswith('stop_head.'))
        optimizer_counts[phase] = int(max(float(record['step']) for record in payload['optimizer']['state'].values()))
    train_ids = set(manifest['train'])
    heldout_ids = set(manifest['teacher_heldout'])
    internal_ids = set(manifest['internal_selection'])
    checks = {'source_and_old_artifact_integrity': not changed, 'teacher_labels_recomputed': all(label_checks),
              'STOP_parameters_unchanged': all(stop_checks.values()),
              'no_heldout_gradient_samples': not any(heldout_ids & set(payload['sample_ids']) for payload in checkpoints.values()),
              'pilot_selection_labels_not_in_pilot_fit': all(not (internal_ids & set(checkpoints[phase]['sample_ids']))
                                                             for phase in ('task22_reset', 'update14_reset')),
              'final_fit_exactly_credible_train': set(checkpoints['warmup']['sample_ids']) == train_ids ==
                   {sample['anchor_id'] for sample in samples if sample['partition'] == 'teacher_train' and sample['confident']},
              'no_group_or_critic': all(payload['critic'] is None and not payload['ppo_eligible'] for payload in checkpoints.values()),
              'optimizer_steps_match': optimizer_counts == audit['optimizer_updates_by_phase']}
    replay = {}
    stratified = []
    max_error = 0.
    for phase, checkpoint in (('before', output / f"{selection['chosen']}_initial.pt"), ('after', output / 'warmup_last.pt')):
        actor = SpatialQueryActor()
        actor.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['actor'])
        summary, rows = evaluate(actor, samples)
        replay[phase] = summary
        for subset, values in summary.items():
            for key in ('weighted_nll', 'top1', 'camera_hit'):
                max_error = max(max_error, abs(values[key] - audit[phase][subset][key]))
        for dimension in ('j', 'scene', 'state_source'):
            grouped = defaultdict(list)
            for row in rows:
                grouped[(row['partition'], row[dimension])].append(row)
            for (partition, stratum), records in grouped.items():
                for credible in (False, True):
                    subset = [row for row in records if not credible or row['confident']]
                    if not subset:
                        continue
                    stratified.append({'phase': phase, 'partition': partition, 'dimension': dimension, 'stratum': stratum,
                                       'credible_only': credible, 'count': len(subset),
                                       'weighted_nll': float(np.average([row['nll'] for row in subset], weights=[row['weight'] for row in subset])),
                                       'top1': float(np.mean([row['top1'] for row in subset])),
                                       'camera_hit': float(np.mean([row['camera_hit'] for row in subset]))})
    checks['cpu_metric_replay'] = max_error < 1e-4
    steps = list(csv.DictReader((output / 'training.csv').open()))
    checks['finite_training_log'] = len(steps) == audit['optimizer_updates'] and all(np.isfinite(float(row[key])) for row in steps for key in ('loss', 'grad_norm'))
    checks['actor_changed'] = any(not torch.equal(value, initial) for value, initial in zip(
        checkpoints['warmup']['actor'].values(),
        torch.load(output / f"{selection['chosen']}_initial.pt", map_location='cpu', weights_only=True)['actor'].values()))
    expected_early_stop = not (audit['after']['teacher_heldout_confident']['weighted_nll'] < audit['before']['teacher_heldout_confident']['weighted_nll'] and
                               audit['after']['teacher_heldout_confident']['top1'] > audit['before']['teacher_heldout_confident']['top1'])
    progress = json.loads((output / 'PROGRESS_GATE.json').read_text())
    checks['early_stop_honored'] = expected_early_stop and progress['Gate1'] == 'INCONCLUSIVE' and progress['Gate2'] == progress['Gate3'] == 'NOT_RUN'
    report = {'passed': all(checks.values()), 'checks': checks, 'changed_unaccounted_inputs': changed,
              'post_run_source_substitutions': substitutions, 'max_cpu_metric_error': max_error,
              'checkpoint_sha256': sha256(output / 'warmup_last.pt'), 'optimizer_steps': optimizer_counts,
              'initialization_top1_tie_warning': 'uniform camera logits and zero means tie four central candidates; top1 uses first argmax, not tie-adjusted accuracy',
              'inference': 'heldout NLL degradation independently triggers early stop; j/split shift prevents attributing this solely to architecture',
              'verifier_sha256': sha256(__file__)}
    write_csv(output / 'stratified_supervision.csv', stratified)
    write_json(output / 'INDEPENDENT_VERIFICATION.json', report)
    if not report['passed']:
        raise AssertionError(report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    verify(args.output)


if __name__ == '__main__':
    main()
