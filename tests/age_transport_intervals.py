import sys, os, argparse
import numpy as np
from scipy.stats import ks_2samp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import old.causal as c

ap = argparse.ArgumentParser()
ap.add_argument("reps", type=int, default=30)
ap.add_argument("source", choices=["pb", "all"], default="all")
ap.add_argument("width", type=int, default=10, help="age bin width (years)")
ap.add_argument("intervals", nargs="*",
                default=["-2", "-1", "1", "2"])
args = ap.parse_args()

pc = c.load_pc()
X, y, ag, sx, pid, up, gr, tasks = c.load_nv()

import openpyxl
pc_sex = {}
for path in (os.path.join(BASE, 'data/ParkCeleb/PD_demo.xlsx'),
             os.path.join(BASE, 'data/ParkCeleb/CN_demo.xlsx')):
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    hdr = [str(x).strip().lower().replace(" ", "") for x in rows[0]]
    gi = hdr.index("gender")
    for r in rows[1:]:
        if r[0] is None or r[gi] is None:
            continue
        g = str(r[gi]).strip()
        pc_sex[str(r[0]).strip()] = 1.0 if g[:1].lower() == "m" else 0.0

def age_weights_binned(a_src, a_trg, width=10):
    edges = np.arange(20, 110, width)
    hs = np.histogram(a_trg, edges)[0] + 0.5
    ps = np.histogram(a_src, edges)[0] + 0.5
    pt = hs / hs.sum()
    psn = ps / ps.sum()
    b_src = np.clip(np.searchsorted(edges, a_src, side="right") - 1, 0, len(edges) - 2)
    w = pt[b_src] / psn[b_src]
    return w / w.mean()

def age_weights_kde(a_src, a_trg):
    return c._tcb_weights(a_trg, a_src, None)

def sex_weights_binned(s_src, s_trg):
    ps = np.bincount(s_src.astype(int), minlength=2) + 0.5
    pt = np.bincount(s_trg.astype(int), minlength=2) + 0.5
    ps, pt = ps / ps.sum(), pt / pt.sum()
    w = pt[s_src.astype(int)] / ps[s_src.astype(int)]
    return w / w.mean()

def age_sex_weights_binned(a_src, s_src, a_trg, s_trg, width=10):
    edges = np.arange(20, 110, width)
    nb = len(edges) - 1
    ib = lambda a: np.clip(np.searchsorted(edges, a, side="right") - 1, 0, nb - 1)
    hs = np.full((nb, 2), 0.5)
    ht = np.full((nb, 2), 0.5)
    for b, s in zip(ib(a_src), s_src.astype(int)):
        hs[b, s] += 1
    for b, s in zip(ib(a_trg), s_trg.astype(int)):
        ht[b, s] += 1
    hs /= hs.sum()
    ht /= ht.sum()
    w = np.array([ht[b, s] / hs[b, s] for b, s in zip(ib(a_src), s_src.astype(int))])
    return w / w.mean()

def _speaker_ba(sid, y, prob):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob)
    yg, pg = [], []
    for s in np.unique(np.asarray(sid)):
        mm = np.asarray(sid) == s
        if mm.sum() == 0:
            continue
        yg.append(int(np.bincount(y[mm]).argmax()))
        pg.append(prob[mm].mean())
    return c._balanced_acc(np.array(yg), (np.array(pg) >= 0.5).astype(int))

def _ks_dec_gt_conf(dec, conf):
    dec = np.asarray(dec, dtype=float)[np.isfinite(dec)]
    conf = np.asarray(conf, dtype=float)[np.isfinite(conf)]
    if len(dec) < 3 or len(conf) < 3:
        return 1.0
    return float(ks_2samp(dec, conf, alternative="less").pvalue)

def run2(sX, sy, sage, ssx, label):
    mu_s, sd_s = sX.mean(0), np.maximum(sX.std(0), 1e-9)
    Xs = (sX - mu_s) / sd_s
    for n in args.intervals:
        X_pc, y_pc, sid_pc, age_pc = pc[n]
        m0 = ~np.isnan(X_pc).any(axis=1)
        X_t = (X_pc[m0] - mu_s) / sd_s
        y_t, sid_t, age_t = y_pc[m0], sid_pc[m0], age_pc[m0]
        sx_t = np.array([pc_sex.get(s, np.nan) for s in sid_t])
        if np.isnan(sx_t).any():
            print("%-8s %s !! sex missing for some clips, skipping" % (label, n))
            continue
        w_b = age_weights_binned(sage, age_t, width=args.width)
        w_k = age_weights_kde(sage, age_t)
        w_s = sex_weights_binned(ssx, sx_t)
        w_as = age_sex_weights_binned(sage, ssx, age_t, sx_t, width=args.width)
        ess = (w_k.sum() ** 2) / (w_k ** 2).sum()
        conf, arm_b, arm_k, arm_s, arm_as = [], [], [], [], []
        for rep in range(args.reps):
            g_cf = c._gbc(Xs, sy, seed=42 + rep)
            conf.append(_speaker_ba(sid_t, y_t, g_cf.predict_proba(X_t)[:, 1]))
            g = c._gbc(Xs, sy, w_b, seed=42 + rep)
            arm_b.append(_speaker_ba(sid_t, y_t, g.predict_proba(X_t)[:, 1]))
            g = c._gbc(Xs, sy, w_k, seed=42 + rep)
            arm_k.append(_speaker_ba(sid_t, y_t, g.predict_proba(X_t)[:, 1]))
            g = c._gbc(Xs, sy, w_s, seed=42 + rep)
            arm_s.append(_speaker_ba(sid_t, y_t, g.predict_proba(X_t)[:, 1]))
            g = c._gbc(Xs, sy, w_as, seed=42 + rep)
            arm_as.append(_speaker_ba(sid_t, y_t, g.predict_proba(X_t)[:, 1]))
        conf = np.asarray(conf)
        arm_b = np.asarray(arm_b)
        arm_k = np.asarray(arm_k)
        arm_s = np.asarray(arm_s)
        arm_as = np.asarray(arm_as)
        mc = float(np.median(conf))
        print("%-8s %s | conf=%.3f | bins=%.3f (d %+.3f, p=%.3f) | kde=%.3f "
              "(d %+.3f, p=%.3f) | sex=%.3f (d %+.3f, p=%.3f) | agexs=%.3f "
              "(d %+.3f, p=%.3f) | ESS=%.0f%%" % (
                  label, n, mc, np.median(arm_b), np.median(arm_b) - mc,
                  _ks_dec_gt_conf(arm_b, conf), np.median(arm_k),
                  np.median(arm_k) - mc, _ks_dec_gt_conf(arm_k, conf),
                  np.median(arm_s), np.median(arm_s) - mc,
                  _ks_dec_gt_conf(arm_s, conf), np.median(arm_as),
                  np.median(arm_as) - mc, _ks_dec_gt_conf(arm_as, conf),
                  100 * ess / len(sy)), flush=True)

pb = tasks == "PETACA_BLANCA"
if args.source == "pb":
    print("source = NV Petaca Blanca (n=%d, bin width=%dy)  [U-transport dec arm "
          "reference: 0.48-0.51]" % (pb.sum(), args.width), flush=True)
    run2(X[pb], y[pb], ag[pb], sx[pb], "pb")
else:
    print("source = ALL NV pooled (n=%d, bin width=%dy)  [U-transport dec arm "
          "reference: 0.48-0.51]" % (len(y), args.width), flush=True)
    run2(X, y, ag, sx, "all-nv")