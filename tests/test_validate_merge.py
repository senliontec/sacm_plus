"""DDP 分布式验证的 rank-0 汇总逻辑(_merge_val_payloads)回归测试。

核心不变量:
1. 混淆计数跨卡 F1 == 全量拼接后 sklearn F1(逐位等价);
2. nanmean 聚合:单分片 NaN 不传染整列,全 NaN 保持 NaN。
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from core.trainer import _dataset_name_from_image_name, _merge_val_payloads


def _payload(pred, gt, loss_sum=1.5, count=2, cldice=(0.8, np.nan),
             metrics=None, per_dataset=None, empty_sum=0, image_count=2):
    pred_b = (pred > 0)
    gt_b = (gt > 0.5)
    if metrics is None:
        metrics = {'dice': [0.9, np.nan], 'hd': [5.0, 6.0]}
    return {
        'loss_sum': loss_sum, 'count': count,
        'tp': int((pred_b & gt_b).sum()),
        'fp': int((pred_b & ~gt_b).sum()),
        'fn': int((~pred_b & gt_b).sum()),
        'cldice': list(cldice), 'metrics': metrics,
        'per_dataset': per_dataset or {},
        'empty_sum': empty_sum, 'image_count': image_count,
    }


def test_f1_matches_global_sklearn():
    rng = np.random.default_rng(0)
    shards, all_pred, all_gt = [], [], []
    for _ in range(4):
        pred = rng.integers(-2, 2, size=200).astype(float)
        gt = rng.integers(0, 2, size=200).astype(float)
        shards.append(_payload(pred, gt))
        all_pred.append(pred)
        all_gt.append(gt)

    (avg_loss, f1, cldice, metrics, per_dataset,
     empty_rate) = _merge_val_payloads(shards)

    p = (np.concatenate(all_pred) > 0).astype(np.uint8)
    t = (np.concatenate(all_gt) > 0.5).astype(np.uint8)
    assert f1 == pytest.approx(f1_score(t.reshape(-1), p.reshape(-1)))
    assert avg_loss == pytest.approx(1.5 / 2)      # loss_sum/count 的全局均值
    assert cldice == pytest.approx(0.8)            # NaN 被 nanmean 跳过
    assert metrics['dice'] == pytest.approx(0.9)   # 分片间 NaN 不传染
    assert metrics['hd'] == pytest.approx(5.5)
    assert empty_rate == pytest.approx(0.0)


def test_all_nan_keeps_nan():
    g = {'loss_sum': 0.0, 'count': 1, 'tp': 0, 'fp': 0, 'fn': 0,
         'cldice': [np.nan], 'metrics': {'bm': [np.nan]},
         'empty_sum': 0, 'image_count': 1}
    (avg_loss, f1, cldice, metrics, per_dataset,
     empty_rate) = _merge_val_payloads([g])
    assert avg_loss == 0.0
    assert f1 == 0.0                                  # 零分母 -> 0
    assert np.isnan(cldice)
    assert np.isnan(metrics['bm'])
    assert empty_rate == pytest.approx(0.0)


def test_merge_across_uneven_shards():
    # 分片间键不同(引擎列只在 rank 0 存在)也能正确合并
    g0 = _payload(np.zeros(10), np.ones(10), count=1,
                  metrics={'dice': [0.5], 'bm': [2.0]})
    g1 = _payload(np.ones(10), np.ones(10), count=1,
                  metrics={'dice': [1.0]})
    _, _, _, metrics, per_dataset, _ = _merge_val_payloads([g0, g1])
    assert metrics['dice'] == pytest.approx(0.75)
    assert metrics['bm'] == pytest.approx(2.0)        # 只按存在分片平均


def test_empty_shard_contributes_zero():
    # 3-shot 协议层:验证集 < 卡数时,部分 rank 分片为空(0 张图),
    # 空分片贡献 0 计数、不影响 F1 与其他指标
    g_full = _payload(np.ones(10), np.ones(10), count=1, cldice=(0.9,),
                      metrics={'dice': [1.0]})
    g_empty = {'loss_sum': 0.0, 'count': 0, 'tp': 0, 'fp': 0, 'fn': 0,
               'cldice': [], 'metrics': {}, 'empty_sum': 0,
               'image_count': 0}
    (avg_loss, f1, cldice, metrics, per_dataset,
     empty_rate) = _merge_val_payloads([g_full, g_empty])
    assert f1 == pytest.approx(1.0)          # 只有非空分片参与 F1
    assert avg_loss == pytest.approx(1.5)    # 1.5 / 1
    assert cldice == pytest.approx(0.9)
    assert metrics['dice'] == pytest.approx(1.0)
    assert empty_rate == pytest.approx(0.0)  # 空分片不贡献图片数


def test_empty_prediction_rate_merges_globally():
    # Same definition as baseline_zoo.runtime.evaluate: empty images over
    # all validation images, counted across DDP shards.
    g0 = _payload(np.ones(4), np.ones(4), empty_sum=1, image_count=3)
    g1 = _payload(np.ones(4), np.ones(4), empty_sum=2, image_count=3)
    _, _, _, _, _, empty_rate = _merge_val_payloads([g0, g1])
    assert empty_rate == pytest.approx(0.5)


def test_per_dataset_metrics_merge_across_shards():
    g0 = _payload(np.ones(10), np.ones(10), per_dataset={
        'DRIVE': {'dice': [0.5], 'cldice': [0.4]},
    })
    g1 = _payload(np.ones(10), np.ones(10), per_dataset={
        'DRIVE': {'dice': [1.0], 'cldice': [0.8]},
        'CHASEDB1': {'dice': [0.9]},
    })
    _, _, _, _, per_dataset, _ = _merge_val_payloads([g0, g1])
    assert per_dataset['DRIVE']['dice'] == pytest.approx(0.75)
    assert per_dataset['DRIVE']['cldice'] == pytest.approx(0.6)
    assert per_dataset['CHASEDB1']['dice'] == pytest.approx(0.9)


def test_dataset_name_from_image_name():
    assert _dataset_name_from_image_name('CHASEDB1_Image_04L.png') == 'CHASEDB1'
    assert _dataset_name_from_image_name('CrackTree200_train_01.png') == 'CrackTree200'
    assert _dataset_name_from_image_name('DRIVE_01.tif') == 'DRIVE'
