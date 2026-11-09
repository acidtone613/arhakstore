"""Port of AR-HAK's own nine-part verification (tests.py) as pytest.

Checks mirror the executed hypotheses of the RC-340/343/349/352/353 cluster:
  T1  [48,12,16] Superbridge minimum distance >= 16   (RC-340 H1)
  T2  <=7 corrupted atoms exact decode                 (RC-340 H2, RC-338 H4)
  T3  Witt hexads cover all 792 5-subsets              (RC-343 construction)
  T4  Z7 topological mapping, 48 atoms -> 7 groups     (RC-353:83)
  T5  group-mate entropy restoral beats erasure        (RC-353 T5)
  T6  Steiner attention bridge: >=3 shared -> high weight (RC-343 H4)
  T7  23-tick phase gating: FLOPs cut, low fidelity loss (RC-343 H3)
  T8  276-slot hologram (Phi_48 orthonormal)           (RC-353 T4)
  T9  MemoryLayer heal restores content/codes          (RC-340 H8 style)

Imported from the upstream `arhak` package; thresholds are preserved as-is.
"""

import numpy as np
import pytest

from arhakstore.mem import attention as attention_mod
from arhakstore.mem import cores, designs, hologram
from arhakstore.mem.attention import SteinerAttention
from arhakstore.mem.layer import MemoryLayer, five_from_bits
from arhakstore.mem.repair import Repair


def test_t1_superbridge_min_distance():
    mw = cores.min_weight()
    wc = cores.weight_counts()
    assert mw >= 16
    assert wc[16] > 0


def test_t2_byzantine_decode_up_to_7():
    rng = np.random.default_rng(340)
    msgs = rng.integers(0, 2, size=(256, 12)).astype(np.uint8)
    cws = cores.encode(msgs)
    for k in [0, 1, 3, 5, 7]:
        corr = cws.copy()
        if k:
            for t in range(corr.shape[0]):
                cp = rng.choice(48, size=k, replace=False)
                corr[t, cp] = 1 - corr[t, cp]
        dec, valid, _ = cores.decode_nearest(corr, max_dist=7)
        acc = float(np.all(dec == msgs, axis=1).mean() * valid.mean())
        assert acc == 1.0, f"k={k} accuracy {acc}"


def test_t3_witt_hexad_coverage():
    assert len(designs.HEXADS) == 132
    assert len(designs.FIVE_TO_HEXAD) == 792


def test_t4_z7_topological_mapping():
    z7 = designs.atom_to_z7(48)
    bc = np.bincount(z7, minlength=7)
    assert len(bc) == 7
    assert np.all(bc > 0)
    temporal_var = np.var(np.angle(np.exp(2j * np.pi * np.ones(48) / 7)))
    topo_var = np.var(np.angle(np.exp(2j * np.pi * z7 / 7)))
    assert topo_var > temporal_var + 0.5
    for period in [71346, 7590, 46, 759, 253, 1081]:
        assert period % 7 != 0


def test_t5_groupmate_entropy_restoral_beats_erasure():
    rng = np.random.default_rng(353)
    M, N_ATOMS, BLOCK = 400, 48, 2
    z7 = designs.atom_to_z7(48)
    group_base = rng.standard_normal((7, BLOCK))
    values = np.repeat(group_base[z7][None], M, axis=0).reshape(M, N_ATOMS * BLOCK)
    values += 0.4 * rng.standard_normal((M, N_ATOMS * BLOCK))
    repair = Repair(n_atoms=N_ATOMS, block=BLOCK)
    for k in [1, 3, 5, 7, 12]:
        dead = set(rng.choice(N_ATOMS, size=k, replace=False).tolist())
        x_corrupt = values.copy()
        for atom in dead:
            x_corrupt[:, repair.slices(atom)] = 0.0
        healed = repair.groupmate_restore(x_corrupt, dead)
        f_heal = float(np.mean([np.corrcoef(values[i], healed[i])[0, 1]
                                for i in range(M)]))
        f_dead = float(np.mean([np.corrcoef(values[i], x_corrupt[i])[0, 1]
                                for i in range(M)]))
        assert f_heal > f_dead + 1e-4, f"k={k}"
        assert f_heal > 0.9 - 0.02 * k, f"k={k}"


def test_t6_attention_bridge_shared_atoms():
    att = SteinerAttention(seed=343)
    rng = np.random.default_rng(343)
    rows = np.array([len(att.hexad_sets[i] & att.hexad_sets[j])
                     for i, j in rng.integers(0, 132, size=(3000, 2))])
    assert (rows[rows >= 3]).size > 0
    assert all(att.bridge_mask(h).any() for h in range(132))


def test_t7_phase_gating_flops_cut_and_fidelity():
    hexad_mod23 = designs.hexad_mod23()
    full_fid, gated_fid = [], []
    n_hexads = len(designs.HEXADS)
    flops = []
    for tick in range(23):
        phases = (hexad_mod23 * tick) % 23
        resonant = phases == 0
        y = np.random.default_rng(343 + tick).standard_normal(12)
        full_out = np.zeros(12)
        for hx in designs.HEXADS:
            full_out[list(hx)] += y[list(hx)]
        full_out /= n_hexads
        gated_out = np.zeros(12)
        n_active = int(resonant.sum())
        if n_active > 0:
            for i, hx in enumerate(designs.HEXADS):
                if resonant[i]:
                    gated_out[list(hx)] += y[list(hx)]
            gated_out /= n_active
        full_fid.append(np.corrcoef(full_out, y)[0, 1])
        gated_fid.append(np.corrcoef(gated_out, y)[0, 1])
        flops.append(1.0 - n_active / n_hexads)
    mean_flops = float(np.mean(flops))
    loss = max(0.0, float(np.max([full_fid[i] - gated_fid[i] for i in range(23)])))
    assert mean_flops >= 0.90
    assert loss <= 0.05


def test_t8_hologram_orthonormal():
    phi = hologram.build_phi48()
    assert hologram.gram_residual(phi) < 1e-12
    rng = np.random.default_rng(7)
    x = rng.standard_normal(48)
    x = x / np.linalg.norm(x)
    f = hologram.fidelity(x, hologram.decode_state(phi, hologram.encode_state(phi, x)))
    assert f > 0.99999


def test_t9_memory_layer_self_heal(tmp_path):
    layer = MemoryLayer(block=2, seed=343)
    rng = np.random.default_rng(9)
    n_write = 300
    for i in range(n_write):
        kv = rng.standard_normal(48)
        vv = rng.standard_normal(48 * 2)
        layer.write(kv, vv, meta=f"mem-{i}")
    assert layer.stats()["entries"] == n_write
    dead7 = rng.choice(48, size=7, replace=False).tolist()
    layer.kill(dead7)
    assert len(layer.stats()["dead"]) == 7
    heal0 = layer.heal()
    assert heal0["atoms_healed"] == 7
    dec, valid, _ = cores.decode_nearest(layer.codes, max_dist=7)
    codes_ok = all(np.array_equal(layer.codes[i], cores.encode(dec[i]))
                   for i in range(len(dec)))
    assert codes_ok
    sl = slice(dead7[0] * 2, dead7[0] * 2 + 2)
    nonzero_after = np.count_nonzero(layer.values[:, sl])
    assert nonzero_after > 0.8 * n_write
    p = tmp_path / "bank.npz"
    layer.save(str(p))
    layer2 = MemoryLayer(block=2)
    layer2.load(str(p))
    assert layer2.stats()["entries"] == n_write
    np.testing.assert_allclose(layer2.values, layer.values)


def test_attention_module_importable():
    assert attention_mod.SIG_DIM == 46