# refpack

Implementation of sections 1 to 9 of the HedgeFormer work plan.

Every module here exists to close a specific gap that would otherwise cost the
paper a major revision, or to add something that is currently missing from the
literature entirely. Nothing in here is decorative.

## Install and check

```bash
pip install numpy scipy pandas rasterio torch
python selftest.py          # exercises every numeric claim on synthetic data
```

`selftest.py` is not a formality. The coarea length estimator, the achievable
ceiling, the Kulpa correction, the capture-recapture estimators and the
significance machinery are all validated there against cases whose answer is
known in advance. If it does not pass, nothing derived from these modules
belongs in the manuscript.

The modules that need `torch` import it lazily, so the plan, the metrics and
the length estimators all run on a machine with no GPU and no PyTorch.

## What each module does

| Module | Section | Closes |
|---|---|---|
| `config.py` | shared | Class definitions, target-construction constants, the run matrix, and the provisional biomass table |
| `s1_data_audit.py` | 1 | Elevation provenance, DSM against CHM, phenological offset, co-registration, class support in kilometres |
| `s2_protocol.py` | 2 | Equal epochs, one checkpoint criterion, three seeds, nested stratified data fractions, parameter and inference cost |
| `s3_baselines.py` | 3 | Single-modality, early fusion, late fusion, and a classical non-learning baseline |
| `s4_ablation.py` | 4 | Loss ablation, including the textbook clDice that the paper's two corrections are meant to fix |
| `s5_mechanism.py` | 5 | Fast-weight probe, modality ablation at inference, synthetic phenology stress |
| `s6_metrics.py` | 6 | Ordinal metrics, confusion in kilometres, achievable ceiling, gap distribution, landscape connectivity, carbon, orientation, stratification, label-free quality |
| `s7_length.py` | 7 | Threshold-free length from the proximity field, digitisation bias, capture-recapture, positive-unlabelled correction |
| `s8_benchmark.py` | 8 | Feature-tensor export, the invertibility attack, datasheet |
| `s9_run_all.py` | 9 | Phases, run budget, checklist, manuscript tables |
| `stats.py` | 2.3 | Seed aggregation, paired tests, effect sizes, practical significance |

## Start here

```bash
# what to run, in what order, and what it costs
python -m refpack.s9_run_all --emit-scripts

# the data facts that block submission
python -m refpack.s1_data_audit --dem Scene_1_Height.tif --ms scene1.tif \
    --patches ./data/Scene1/npz

# parameter counts and inference cost, fills three FILL entries
python -m refpack.s2_protocol
```

## Things worth knowing before you use this

**The elevation layer.** `inference_patching.py` reads `Scene_2_Height.tif`,
which is a normalised height layer rather than absolute elevation, and
`read_ms_on_dem_grid` resamples the multispectral stack bilinearly onto the
1 m elevation grid. Both facts belong in the paper and neither is currently
stated. `s1_data_audit.classify_elevation_layer` confirms which product you
actually have from the value distribution.

**The match rate already exists.** `validation.FullMetrics` computes
`matched_pred_comp / num_pred_comp`, reported as connectivity correctness.
That is the instance match rate. It needs to appear next to every Chamfer and
Frechet figure in the paper, because matched-only geometry improves for free
when a model detects fewer and easier instances, and that is the sharpest
attack available against the current geometric claim.

**Significance with thousands of patches.** A paired test over several
thousand patches will return a tiny p-value for a difference in the fourth
decimal place. `stats.compare_variants` therefore takes a `min_effect` and
reports a practical verdict alongside the statistical one. Set it to the
smallest difference that would change a downstream decision.

**The biomass coefficients are placeholders.** `config.BIOMASS_T_PER_KM` is
flagged `PROVISIONAL` and exists so the uncertainty propagation can be
exercised. Replace every row with a sourced value and record the source before
any carbon number appears in the manuscript. `carbon_stock` and
`binary_map_carbon_bias` both return
`coefficients_are_provisional` so the flag travels with the result.

**Capture-recapture can blow up.** `chao_estimator` returns `unstable: True`
when singletons dominate, which happens when detectors fragment the same
hedges differently rather than missing different hedges. Raise the match
tolerance until whole hedges merge into one segment, or the estimate is
meaningless. The self-test demonstrates both regimes.

**The achievable ceiling is optimistic.** It assumes reference errors are
independent across pixels. Real errors are spatially correlated, so the true
ceiling is lower. `achievable_ceiling` returns that assumption in its output
so it cannot be quoted without it.

## Three things to do first if time is short

1. `s5_mechanism.modality_ablation`. No retraining, and it proves the
   mechanism directly instead of inferring it from a score. Look for the
   cross-over: the adaptive operator should lose least in class 1a when the
   elevation stream is destroyed and most in class 6.
2. `s6_metrics.ordinal_metrics`. Height error in metres is the number that
   makes these results legible next to the canopy-height literature, where
   errors run to five and nine metres.
3. `s6_metrics.achievable_ceiling`. A general tool other groups will borrow,
   and the permanent answer to a reviewer objecting to an absolute score of
   0.43 against a reference that is itself about one third correct.
