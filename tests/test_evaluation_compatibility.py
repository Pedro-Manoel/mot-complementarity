"""Reuse of metrics: intact products of another corpus or protocol are refused."""
import json

import pytest

from pedestrian_catalog import detection as D, evaluation as EV
from pedestrian_catalog.provenance import sha256_file
from pedestrian_catalog.schema import stable_id


@pytest.fixture
def products(tmp_path):
    c = {'config_id': 'c', 'config_hash': 'config', 'cache_key': 'cache',
         'detector_variant_id': 'v', 'checkpoint': {'sha256': 'p' * 64}}
    s = {'dataset': 'MOT20', 'sequence_id': 'S', 'fps': 30., 'n_frames': 1,
         'frames_manifest_sha256': 'f' * 64, 'annotation_protocol': 'motchallenge_mot20',
         'gt_sha256': 'g' * 64, 'gt_rows_after_last_frame': 0}
    di = D.detection_cell_identity(c, s, 42, {}, {'device': 'cpu'})
    dc = D.cell_dir(tmp_path, c, s, 42)
    dc.mkdir(parents=True)
    dm = {'cell_id': di['cell_id'], 'identity': {k:v for k,v in di.items() if k!='cell_id'},
          'cell_dir': str(dc), 'status': 'complete'}
    (dc/'manifest.json').write_text(json.dumps(dm), encoding='utf-8')
    ti = {'stage': 'tracking', 'config_hash': c['config_hash'], 'config_id': c['config_id'],
          'detection_cell_id': dm['cell_id'], 'fps': 30.}
    tc = EV._tracks_manifest_path(tmp_path,c,s,42).parent
    tc.mkdir(parents=True)
    mot = tc/'tracks_mot.txt'
    mot.write_text('1,1,0,0,10,10,1,-1,-1,-1\n',encoding='utf-8')
    tm = {'cell_id': stable_id(ti), 'identity': ti, 'cell_dir': str(tc), 'status': 'complete',
          'config_id': 'c', 'sequence_key': 'MOT20/S', 'replicate_seed': 42,
          'detection_cell_id': dm['cell_id'], 'fps': 30.,
          'files': {'tracks_mot.txt': {'sha256': sha256_file(mot)}}}
    (tc/'manifest.json').write_text(json.dumps(tm),encoding='utf-8')
    ei = EV.evaluation_cell_identity(s,tm,'MOT20',True,{'trackeval_commit': 'tested'})
    ec = EV.cell_dir(tmp_path,tm,ei)
    ec.mkdir(parents=True)
    met = ec/'metrics.json'
    met.write_text(json.dumps({'normalized': {'HOTA': 80.}}),encoding='utf-8')
    em = {**ei,'identity':{k:v for k,v in ei.items() if k!='cell_id'}, 'cell_dir':str(ec),
          'files':{'metrics.json': {'sha256': sha256_file(met)}}}
    (ec/'manifest.json').write_text(json.dumps(em),encoding='utf-8')
    return tmp_path,c,s


def test_collection_keeps_a_compatible_evaluation(products):
    run,c,s = products
    rows,issues = EV.collect_metrics(run,[c],[s],[42],True)
    assert issues == [] and len(rows)==1 and rows[0]['HOTA']==80.


@pytest.mark.parametrize('field,value', [('fps',60.),('frames_manifest_sha256','0'*64),
                                       ('annotation_protocol','personpath22_keyframes')])
def test_collection_refuses_a_different_corpus_or_protocol(products,field,value):
    run,c,s = products
    other = dict(s,**{field:value})
    rows,issues = EV.collect_metrics(run,[c],[other],[42],True)
    assert rows == [] and len(issues)==1


@pytest.mark.parametrize('field,value', [('fps',60.),('frames_manifest_sha256','0'*64)])
def test_evaluation_command_refuses_incompatible_tracks_before_the_evaluator(products,monkeypatch,field,value):
    run,c,s = products
    def must_not_evaluate(*args,**kwargs):
        pytest.fail('incompatible tracks reached TrackEval')
    monkeypatch.setattr(EV,'evaluate_batch',must_not_evaluate)
    coverage = EV.run_evaluation([c],[dict(s,**{field:value})],[42],run,evaluator={})
    assert coverage['complete']==[] and len(coverage['failed'])==1


def test_collection_refuses_a_tampered_detection_manifest(products):
    run,c,s = products
    p = D.cell_dir(run,c,s,42)/'manifest.json'
    m = json.loads(p.read_text(encoding='utf-8'))
    m['identity']['sequence']['frames_manifest_sha256']='0'*64
    p.write_text(json.dumps(m),encoding='utf-8')
    rows,issues = EV.collect_metrics(run,[c],[s],[42],True)
    assert rows==[] and len(issues)==1


def test_collection_refuses_a_tampered_mot_file(products):
    run,c,s = products
    (EV._tracks_manifest_path(run,c,s,42).parent/'tracks_mot.txt').write_text('',encoding='utf-8')
    rows,issues = EV.collect_metrics(run,[c],[s],[42],True)
    assert rows==[] and len(issues)==1
