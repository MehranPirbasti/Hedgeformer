"""
Self-test for the parts of the package that need no GPU and no rasters.

Every numeric claim the package makes is exercised here against data whose
answer is known in advance. If this script does not pass, nothing derived from
these modules belongs in the manuscript.

Run:  python selftest.py
"""
import numpy as np
from scipy import ndimage as ndi

from refpack import s6_metrics as M
from refpack import s7_length as L
from refpack import stats as S
from refpack import s9_run_all as R
from refpack.config import CLASS_CODE, DIST_CAP_M

FAIL = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}  {detail}")
    if not condition:
        FAIL.append(name)


# --------------------------------------------------------------------------
print("\n== 6.1 ordinal metrics")
rng = np.random.default_rng(0)
truth = rng.integers(2, 7, 50_000)   # keep away from the ends so
                                     # that clipping cannot create
                                     # accidental exact matches
perfect = M.ordinal_metrics(truth, truth)
check("perfect prediction is exact", abs(perfect["exact"] - 1.0) < 1e-9)
check("perfect prediction has zero metre error", perfect["mae_metres"] == 0.0)
check("perfect prediction has kappa 1", abs(perfect["qwk"] - 1.0) < 1e-6,
      f"qwk={perfect['qwk']:.6f}")

off_by_one = np.clip(truth + rng.choice([-1, 1], truth.size), 1, 7)
o = M.ordinal_metrics(off_by_one, truth)
check("off-by-one scores zero exact", o["exact"] < 0.02, f"exact={o['exact']:.4f}")
check("off-by-one is fully within one", o["within_one"] > 0.98,
      f"within_one={o['within_one']:.4f}")
check("off-by-one metre error is sub-metre-ish", 0.3 < o["mae_metres"] < 1.6,
      f"mae={o['mae_metres']:.3f} m")
check("kappa penalises but does not collapse", 0.5 < o["qwk"] < 0.95,
      f"qwk={o['qwk']:.4f}")

off_by_three = np.clip(truth + 3, 1, 7)
o3 = M.ordinal_metrics(off_by_three, truth)
check("larger offset gives lower kappa", o3["qwk"] < o["qwk"],
      f"{o3['qwk']:.3f} < {o['qwk']:.3f}")
check("signed bias detects systematic overestimate", o3["bias_metres"] > 1.0,
      f"bias={o3['bias_metres']:.3f} m")

# --------------------------------------------------------------------------
print("\n== 6.3 achievable ceiling")
built = M.build_reference_confusion()
C = built["matrix"]
check("confusion rows are normalised", np.allclose(C.sum(1), 1.0))
check("diagonal reproduces published exact agreement",
      abs(built["achieved_exact"] - 0.32) < 0.01,
      f"exact={built['achieved_exact']:.4f}")
print(f"      within-one achieved {built['achieved_within_one']:.3f} "
      f"against published {built['target_within_one']:.2f}, "
      f"residual {built['within_one_residual']:+.3f}")

prior = np.array([0.10, 0.14, 0.16, 0.22, 0.18, 0.13, 0.07])
ceiling = M.achievable_ceiling(
    prior, score_fn=lambda p, l: M.ordinal_metrics(p, l)["exact"],
    n_pixels=20_000, n_trials=40)
check("ceiling is well below one", 0.2 < ceiling["ceiling_mean"] < 0.45,
      f"ceiling={ceiling['ceiling_mean']:.4f}")
frac = M.fraction_of_ceiling(0.27, ceiling)
print("      headline:", frac["headline"])

# --------------------------------------------------------------------------
print("\n== 7.1 coarea length estimator")
val = L.validate_coarea_estimator()
for case in ("straight", "diagonal", "grid_with_junctions"):
    r = val[case]["ratio"]
    print(f"      {case:22s} true={val[case]['true_px']:8.0f} "
          f"est={val[case]['est_px']:8.0f}  ratio={r:.3f}")
check("straight lines recovered within 10 percent",
      abs(val["straight"]["ratio"] - 1.0) < 0.10)
check("diagonals recovered within 15 percent",
      abs(val["diagonal"]["ratio"] - 1.0) < 0.15)
check("junction correction is a sane factor",
      0.5 < val["recommended_intersection_correction"] < 1.5,
      f"correction={val['recommended_intersection_correction']:.3f}")

# --------------------------------------------------------------------------
print("\n== 7.2 Kulpa correction")
sk = np.zeros((300, 300), bool)
sk[150, 20:280] = True                       # pure horizontal, 259 steps
k = L.kulpa_length(sk)
check("horizontal line barely corrected",
      abs(k["length_px_kulpa"] - 259 * 0.948) < 1.0,
      f"kulpa={k['length_px_kulpa']:.1f}")

sk = np.zeros((300, 300), bool)
rr = np.arange(20, 280)
sk[rr, rr] = True                            # pure diagonal
k = L.kulpa_length(sk)
raw = k["length_px_raw"]
check("diagonal bias is removed", 0 < k["bias_removed_percent"] < 12,
      f"removed {k['bias_removed_percent']:.2f} percent of {raw:.1f}")

# --------------------------------------------------------------------------
print("\n== 7.3 capture-recapture")
truth_mask = np.zeros((400, 400), bool)
for r in (60, 140, 220, 300):
    truth_mask[r, 30:370] = True
for c in (80, 200, 320):
    truth_mask[30:370, c] = True

rng = np.random.default_rng(1)


def partial_detector(mask, keep_frac, jitter=0, seg_len=40):
    """Drops whole segments, which is how a detector actually fails.

    Dropping individual pixels instead would shatter every hedge into
    fragments, and the capture-recapture estimator would then be measuring
    disagreement about fragmentation rather than disagreement about which
    hedges exist. That distinction matters on the real data too.
    """
    out = np.zeros_like(mask)
    lbl, n = ndi.label(mask, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        order = np.argsort(xs if np.ptp(xs) >= np.ptp(ys) else ys)
        ys, xs = ys[order], xs[order]
        for start in range(0, ys.size, seg_len):
            if rng.random() < keep_frac:
                sl = slice(start, start + seg_len)
                out[ys[sl], xs[sl]] = True
    if jitter:
        out = np.roll(out, jitter, axis=0)
    return out


ref = partial_detector(truth_mask, 0.55)
preds = {"ttt": partial_detector(truth_mask, 0.65, 1),
         "sra": partial_detector(truth_mask, 0.60, -1),
         "concat": partial_detector(truth_mask, 0.70)}

rep = L.network_length_report(ref, preds, tolerance_px=3)
true_km = L.kulpa_length(truth_mask)["length_km"]
print(f"      true network      {true_km:.3f} km")
print(f"      reference sees    {rep['reference_length_km']:.3f} km")
for n, v in rep["predicted_length_km"].items():
    print(f"      {n:8s} sees     {v:.3f} km")
if rep["estimated_true_network_km"]:
    print(f"      Chao estimate     {rep['estimated_true_network_km']:.3f} km")
check("two-sample estimate exceeds either single view",
      all(v["estimate"] > rep["reference_length_km"]
          for v in rep["two_sample_estimates_km"].values()
          if np.isfinite(v["estimate"])))
lp = rep["two_sample_estimates_km"]["ttt"]["estimate"]
check("two-sample estimate is within a factor of two of truth",
      0.5 * true_km < lp < 2.0 * true_km, f"LP={lp:.3f} vs true {true_km:.3f} km")
check("Chao reports whether it is stable",
      "unstable" in rep["multi_sample_chao"],
      f"unstable={rep['multi_sample_chao'].get('unstable')}")

h = np.array([1] * 40 + [2] * 18 + [3] * 9 + [4] * 4)
chao = L.chao_estimator(h)
check("Chao estimate exceeds observed",
      chao["estimated_total_segments"] > chao["observed_segments"],
      f"{chao['estimated_total_segments']:.1f} > {chao['observed_segments']}")
check("Chao completeness is a fraction", 0 < chao["completeness"] <= 1,
      f"completeness={chao['completeness']:.3f}")

# --------------------------------------------------------------------------
print("\n== 7.4 positive-unlabelled correction")
pu = L.pu_corrected_rates(precision_obs=0.37, recall_obs=0.51, c=0.70)
check("corrected recall exceeds observed", pu["recall_corrected"] > 0.51,
      f"{pu['recall_corrected']:.4f}")
check("corrected precision exceeds observed", pu["precision_corrected"] > 0.37,
      f"{pu['precision_corrected']:.4f}")

# --------------------------------------------------------------------------
print("\n== 6.2 confusion in kilometres")
true_cls = np.zeros((400, 400), np.uint8)
pred_cls = np.zeros((400, 400), np.uint8)
skel = np.zeros((400, 400), bool)
for i, r in enumerate((60, 140, 220, 300)):
    true_cls[r, 30:370] = i + 2
    pred_cls[r, 30:370] = i + 2
    skel[r, 30:370] = True
pred_cls[140, 30:200] = 5                     # a stretch misassigned
cm = M.confusion_km(pred_cls, true_cls, skel)
check("row sums equal reference length",
      abs(sum(cm["reference_length_km"]) - cm["total_reference_km"]) < 1e-9)
check("exact length is below total", cm["exact_km"] < cm["total_reference_km"],
      f"{cm['exact_km']:.3f} of {cm['total_reference_km']:.3f} km")

# --------------------------------------------------------------------------
print("\n== 6.4 gap distribution")
# the reference itself must contain some gaps, otherwise there is nothing to
# compare the predicted gap distribution against
full = np.zeros((200, 200), bool)
for r in (60, 100, 140):
    full[r, 10:190] = True
for r in (60, 100, 140):
    for c in (70, 130):
        full[r, c:c + 8] = False
broken = full.copy()
for r in (60, 100, 140):
    for c in range(20, 185, 12):
        broken[r, c:c + 3] = False
g = M.gap_distribution_test(broken, full)
check("fragmentation is detected", g["excess_short_gaps"] > 0,
      f"excess={g['excess_short_gaps']:.3f}, p={g['ks_p_value']:.3g}")

# --------------------------------------------------------------------------
print("\n== 6.5 connectivity")
intact = np.zeros((400, 400), bool)
intact[200, 20:380] = True
frag = intact.copy()
for c in range(60, 380, 40):
    frag[200, c:c + 12] = False
ret = M.connectivity_retention(frag, intact, dispersal_m=50.0)
check("fragmentation lowers IIC retention", ret["IIC_retention"] < 1.0,
      f"IIC retention={ret['IIC_retention']:.3f}")
check("fragmentation lowers PC retention", ret["PC_retention"] < 1.0,
      f"PC retention={ret['PC_retention']:.3f}")

# --------------------------------------------------------------------------
print("\n== 6.6 carbon")
length_km = {"1a": 12.0, "1b": 18.0, "1c": 21.0, "2": 30.0,
             "3": 24.0, "4": 15.0, "6": 6.0}
cs = M.carbon_stock(length_km)
check("carbon interval brackets the mean",
      cs["carbon_t"]["ci95"][0] < cs["carbon_t"]["mean"] < cs["carbon_t"]["ci95"][1],
      f"{cs['carbon_t']['mean']:.0f} t C")
bias = M.binary_map_carbon_bias(length_km)
print(f"      height-resolved {bias['height_resolved_carbon_t']:.0f} t C, "
      f"binary map {bias['binary_map_carbon_t']:.0f} t C, "
      f"bias {bias['relative_bias_percent']:+.1f} percent")
check("binary-map bias is flagged as provisional",
      bias["coefficients_are_provisional"] is True)

# --------------------------------------------------------------------------
print("\n== 6.7 orientation")
a = np.zeros((300, 300), bool)
a[:, 150] = True                              # vertical
b = np.zeros((300, 300), bool)
b[150, :] = True                              # horizontal
ot = M.orientation_test(a, b)
check("orthogonal networks are distinguished",
      ot["reject_same_distribution_at_5pct"],
      f"U2={ot['watson_u2']:.3f}")

# --------------------------------------------------------------------------
print("\n== stats")
rng = np.random.default_rng(2)
base = rng.normal(0.43, 0.05, 2000)
better = base + rng.normal(0.004, 0.02, 2000)
cmp_ = S.compare_variants({"ttt": better, "concat": base}, reference="concat",
                          min_effect=0.002)
r = cmp_["ttt"]
print(f"      mean diff {r['mean_diff']:+.5f}  "
      f"CI [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]  "
      f"p={r['p_value']:.3g}  delta={r['cliffs_delta']:+.3f} ({r['effect']})")
check("a real small effect is detected", r["verdict"] == "better")

# a difference far below any meaningful threshold. With two thousand paired
# samples the p-value may still be small, which is exactly why the package
# reports a practical verdict alongside the statistical one.
tiny = base + rng.normal(0.0004, 0.02, 2000)
cmp2 = S.compare_variants({"x": tiny, "concat": base}, reference="concat",
                          min_effect=0.005)
print(f"      tiny effect: diff {cmp2['x']['mean_diff']:+.5f}, "
      f"p={cmp2['x']['p_value']:.3g}, delta={cmp2['x']['cliffs_delta']:+.3f}")
check("a negligible difference is not claimed as a win",
      not cmp2["x"]["exceeds_min_effect"],
      cmp2["x"]["practical_verdict"])
check("effect size is reported as negligible",
      S.interpret_delta(cmp2["x"]["cliffs_delta"]) == "negligible")

summ = S.summarise_seeds({"TTT": {"macro_f1": [0.4380, 0.4361, 0.4374]},
                          "Simple": {"macro_f1": [0.4377, 0.4342, 0.4359]}})
print("      seed cell:", summ["macro_f1"]["TTT"].cell())
check("seed summary carries a spread", summ["macro_f1"]["TTT"].std > 0)

# --------------------------------------------------------------------------
print("\n== 9 plan")
p = R.plan()
b = R.budget_report()
print(f"      {p['n_tasks']} tasks, {p['training_runs']} training runs full, "
      f"{b['minimal_defensible_runs']} minimal")
check("blocking tasks are identified", len(p["blocking"]) >= 6,
      f"{len(p['blocking'])} blocking")

# --------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAIL:
    print(f"{len(FAIL)} FAILED: {', '.join(FAIL)}")
    raise SystemExit(1)
print("all checks passed")
