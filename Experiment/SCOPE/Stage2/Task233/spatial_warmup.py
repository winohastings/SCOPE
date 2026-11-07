from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from Experiment.SCOPE.Stage2.Task22.actor import ActorState, SpatialQueryActor
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save
from Experiment.SCOPE.Stage2.Task233.analyze import write_csv


def spatial_log_prob(distribution, camera, latent):
    index = torch.tensor(camera, device=distribution.camera.logits.device)
    return distribution.camera.log_prob(index) + distribution.position.log_prob(latent.expand(4, -1))[camera].sum()


def reset_spatial(actor):
    with torch.no_grad():
        for head in (actor.camera_head, actor.position_head):
            head[-1].weight.zero_()
            head[-1].bias.zero_()
        actor.log_std.fill_(math.log(.8))
    actor.stop_head.requires_grad_(False)
    return actor


def load_samples(dataset, device):
    anchors = json.loads((dataset / 'ANCHOR_MANIFEST_256.json').read_text())['anchors']
    quality = {row['anchor_id']: row for row in csv.DictReader((dataset / 'candidate_quality_256.csv').open())}
    groups = defaultdict(dict)
    for path in {row['evidence'] for row in anchors}:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            groups[row['anchor_id']][row['variant']] = row
    samples = []
    for anchor in anchors:
        if sha256(anchor['state_path']) != anchor['state_sha256']:
            raise ValueError('state hash changed')
        payload = torch.load(anchor['state_path'], map_location='cpu', weights_only=True)
        if payload['data_kind'] != 'teacher_candidate' or payload['ppo_eligible']:
            raise ValueError('data-kind mismatch')
        state = ActorState.unpack(payload['state'], device)
        records = [groups[anchor['anchor_id']][f'candidate_{index}'] for index in range(8)]
        if any(row['split'] != 'train' or not row['restoration_valid'] or row['ppo_eligible'] for row in records):
            raise ValueError('split or restoration violation')
        teacher = max(range(8), key=lambda index: records[index]['score_gt'])
        targets = [row['diagnostic']['selected'][-1] for row in records]
        label = quality[anchor['anchor_id']]
        samples.append(dict(anchor, state=state, targets=targets, teacher=teacher,
                            confident=label['confident'] == 'True', weight=min(1., float(label['gt_margin']) / .01)))
    routes = {part: {(row['scene'], str(row['trajectory_id'])) for row in samples if row['partition'] == part}
              for part in ('teacher_train', 'teacher_heldout')}
    if routes['teacher_train'] & routes['teacher_heldout']:
        raise ValueError('train/heldout trajectory leakage')
    return samples


def sample_loss(actor, sample):
    target = sample['targets'][sample['teacher']]
    distribution = actor(sample['state'])
    latent = torch.tensor(target['latent'], device=sample['state'].glance.device)
    return -spatial_log_prob(distribution, target['camera'], latent)


@torch.no_grad()
def evaluate(actor, samples):
    actor.eval()
    rows = []
    for sample in samples:
        distribution = actor(sample['state'])
        scores = [float(spatial_log_prob(distribution, target['camera'],
                  torch.tensor(target['latent'], device=sample['state'].glance.device))) for target in sample['targets']]
        target = sample['targets'][sample['teacher']]
        predicted_camera = int(distribution.camera.probs.argmax())
        rows.append({key: sample[key] for key in ('anchor_id', 'partition', 'scene', 'state_source', 'j', 'confident', 'weight')} |
                    {'nll': -scores[sample['teacher']], 'top1': int(np.argmax(scores) == sample['teacher']),
                     'camera_hit': int(predicted_camera == target['camera']),
                     'latent_error': float(torch.linalg.vector_norm(distribution.position.loc[target['camera']] -
                                           torch.tensor(target['latent'], device=sample['state'].glance.device))),
                     'camera': predicted_camera, 'camera_probs': distribution.camera.probs.cpu().tolist(),
                     'p_stop_diagnostic_only': float(distribution.stop.probs)})
    summaries = {}
    for part in sorted({row['partition'] for row in rows}):
        for credible in (False, True):
            selected = [row for row in rows if row['partition'] == part and (not credible or row['confident'])]
            if not selected:
                continue
            weights = np.asarray([row['weight'] for row in selected])
            summaries[part + ('_confident' if credible else '_all')] = {
                'count': len(selected), 'weighted_nll': float(np.average([row['nll'] for row in selected], weights=weights)),
                'top1': float(np.mean([row['top1'] for row in selected])),
                'camera_hit': float(np.mean([row['camera_hit'] for row in selected]))}
    return summaries, rows


def train(actor, samples, epochs, label, output, seed=233301):
    if not samples or any(sample['partition'] != 'teacher_train' or not sample['confident'] for sample in samples):
        raise ValueError('only credible train labels may receive gradients')
    actor.train()
    actor.stop_head.requires_grad_(False)
    frozen_stop = copy.deepcopy(actor.stop_head.state_dict())
    optimizer = torch.optim.AdamW([parameter for parameter in actor.parameters() if parameter.requires_grad], lr=1e-4, weight_decay=0.)
    generator = random.Random(seed)
    updates = 0
    for epoch in range(epochs):
        order = list(samples)
        generator.shuffle(order)
        loss_values = []
        for offset in range(0, len(order), 8):
            batch = order[offset:offset + 8]
            optimizer.zero_grad(set_to_none=True)
            loss = sum(sample_loss(actor, sample) * sample['weight'] for sample in batch) / sum(sample['weight'] for sample in batch)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite spatial loss')
            loss.backward()
            if any(parameter.grad is not None for parameter in actor.stop_head.parameters()):
                raise ValueError('STOP gradient forbidden')
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            loss_values.append(float(loss.detach()))
            with (output / 'training.csv').open('a') as stream:
                stream.write(f'{label},{epoch + 1},{updates},{float(loss.detach())},{float(norm)}\n')
        if any(not torch.equal(value, frozen_stop[name]) for name, value in actor.stop_head.state_dict().items()):
            raise ValueError('STOP parameter changed')
        atomic_save(output / f'{label}_last.pt', {'actor': actor.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1, 'updates': updates, 'seed': seed, 'data_kind': 'teacher_spatial_bc',
                    'ppo_eligible': False, 'critic': None, 'sample_ids': [sample['anchor_id'] for sample in samples],
                    'python_rng': generator.getstate(), 'torch_rng': torch.get_rng_state()})
        print(json.dumps({'phase': label, 'epoch': epoch + 1, 'updates': updates, 'mean_batch_loss': float(np.mean(loss_values))}), flush=True)
    return updates


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    dataset = args.dataset.resolve()
    parent = ROOT / 'artifacts/gate0_v3'
    audit = json.loads((parent / 'teacher_signal_audit.json').read_text())
    verification = json.loads((dataset / 'EXPANSION_VERIFICATION.json').read_text())
    if audit['status'] != 'PASS' or not verification['passed'] or verification['manifest_sha256'] != sha256(dataset / 'ANCHOR_MANIFEST_256.json'):
        raise ValueError('Gate0 or expanded evidence not verified')
    init_path = ROOT.parent / 'Task22/artifacts/final_runtime/actor_initial.pt'
    update_path = ROOT.parent / 'Task23/artifacts/ppo4096/runtime/checkpoints/update_0014.pt'
    frozen_inputs = {path: expected for path, expected in json.loads((dataset / 'SOURCE_INTEGRITY_BEFORE.json').read_text()).items()}
    frozen_inputs.update({str(path): sha256(path) for path in (init_path, ROOT / 'GATE1_PROTOCOL.md', Path(__file__))})
    if any(sha256(path) != expected for path, expected in frozen_inputs.items()):
        raise ValueError('upstream inputs changed')
    write_json(args.output / 'INPUT_HASHES.json', frozen_inputs)
    write_json(args.output / 'RUN_CONFIG.json', {'seed': 233301, 'pilot_epochs': 3, 'epochs': 12, 'batch_size': 8,
               'lr': 1e-4, 'STOP_loss': False, 'STOP_execution': 'bypassed in future fixed N2/N4 environment evaluation',
               'device': args.device, 'protocol_sha256': sha256(ROOT / 'GATE1_PROTOCOL.md'),
               'dataset': str(dataset), 'critic_training': False, 'heldout_used_for_gradients': False})
    samples = load_samples(dataset, args.device)
    train_samples = [sample for sample in samples if sample['partition'] == 'teacher_train' and sample['confident']]
    routes = sorted({(sample['scene'], str(sample['trajectory_id'])) for sample in train_samples})
    random.Random(233301).shuffle(routes)
    internal_routes = set(routes[:max(1, len(routes) // 5)])
    internal = [sample for sample in train_samples if (sample['scene'], str(sample['trajectory_id'])) in internal_routes]
    pilot_train = [sample for sample in train_samples if (sample['scene'], str(sample['trajectory_id'])) not in internal_routes]
    write_json(args.output / 'TRAIN_MANIFEST.json', {'train': [sample['anchor_id'] for sample in train_samples],
               'internal_selection': [sample['anchor_id'] for sample in internal],
               'teacher_heldout': [sample['anchor_id'] for sample in samples if sample['partition'] == 'teacher_heldout']})
    initial = torch.load(init_path, map_location='cpu', weights_only=False)
    updated = torch.load(update_path, map_location='cpu', weights_only=False)
    if initial['provenance']['backbone_sha256'] != updated['provenance']['backbone_sha256']:
        raise ValueError('initial backbone provenance mismatch')
    states = {'task22_reset': initial['actor'], 'update14_reset': updated['trainer']['old_actor']}
    (args.output / 'training.csv').write_text('phase,epoch,update,loss,grad_norm\n')
    scores, updates = {}, {}
    for label, weights in states.items():
        torch.manual_seed(233301)
        actor = SpatialQueryActor().to(args.device)
        actor.load_state_dict(weights)
        reset_spatial(actor)
        atomic_save(args.output / f'{label}_initial.pt', {'actor': actor.state_dict()})
        updates[label] = train(actor, pilot_train, 3, label, args.output)
        with torch.no_grad():
            scores[label] = sum(float(sample_loss(actor, sample)) * sample['weight'] for sample in internal) / sum(sample['weight'] for sample in internal)
    chosen = min(scores, key=scores.get)
    write_json(args.output / 'INITIALIZATION_SELECTION.json', {'internal_nll': scores, 'chosen': chosen,
               'external_heldout_used': False, 'pilot_updates': updates})
    actor = SpatialQueryActor().to(args.device)
    actor.load_state_dict(torch.load(args.output / f'{chosen}_initial.pt', map_location=args.device, weights_only=True)['actor'])
    before, before_rows = evaluate(actor, samples)
    write_csv(args.output / 'before_predictions.csv', before_rows)
    updates['warmup'] = train(actor, train_samples, 12, 'warmup', args.output)
    after, after_rows = evaluate(actor, samples)
    write_csv(args.output / 'after_predictions.csv', after_rows)
    key = 'teacher_heldout_confident'
    checks = {'heldout_nll_decreased': after[key]['weighted_nll'] < before[key]['weighted_nll'],
              'heldout_candidate_top1_improved': after[key]['top1'] > before[key]['top1'],
              'sources_unchanged': all(sha256(path) == expected for path, expected in frozen_inputs.items())}
    ready = all(checks.values())
    result = {'offline_check': 'PASS' if ready else 'INCONCLUSIVE', 'checks': checks, 'before': before, 'after': after,
              'optimizer_updates_by_phase': updates, 'optimizer_updates': sum(updates.values()), 'wall_seconds': time.monotonic() - began,
              'Gate1': 'NOT_RUN' if ready else 'INCONCLUSIVE', 'requires_real_navigation_acceptance': True,
              'SR_SPL': 'NOT_RUN', 'heldout_critic': 'NOT_RUN: no Critic used', 'new_teacher_calls': 0,
              'reason': 'offline check passed; DAgger and real navigation still required' if ready else 'offline heldout learning signal did not improve under preregistered budget; stop downstream'}
    write_json(args.output / 'OFFLINE_AUDIT.json', result)
    write_json(args.output / 'PROGRESS_GATE.json', {'Gate0': 'PASS', 'Gate1': result['Gate1'], 'Gate2': 'NOT_RUN',
               'Gate3': 'NOT_RUN', 'downstream_locked': True, 'reason': result['reason']})
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError('refuse to overwrite an existing run: ' + str(args.output))
    try:
        run(args)
    except BaseException as error:
        if args.output.exists():
            write_json(args.output / 'INCOMPLETE.json', {'error': repr(error)})
            write_json(args.output / 'PROGRESS_GATE.json', {'Gate0': 'PASS', 'Gate1': 'FAIL', 'Gate2': 'NOT_RUN',
                       'Gate3': 'NOT_RUN', 'downstream_locked': True, 'reason': repr(error)})
        raise


if __name__ == '__main__':
    main()
