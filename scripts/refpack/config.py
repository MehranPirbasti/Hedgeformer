"""
Shared constants for the HedgeFormer reference package.

Every value here is either taken from the existing pipeline or flagged as
PROVISIONAL. Anything marked PROVISIONAL must be replaced with a sourced value
before it goes into the manuscript.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# geometry and target construction, taken from dataprep.py / dataset.py
# --------------------------------------------------------------------------
PIXEL_SIZE_M = 1.0        # the DEM grid is the reference grid, 1 m
DIST_CAP_M = 10.0         # dataprep.distance_cap_meters, also merging dist_cap_m
BUFFER_M = 10.0           # dataprep.buffer_meters
PATCH = 256
TILE_OVERLAP = 64
SEG_THRESH = 0.30         # validation.CFG seg_thresh
METRIC_TOL_PX = 3         # relaxed tolerance
BF2_TOL_PX = 2

# --------------------------------------------------------------------------
# UKCEH Land Cover Plus: Hedgerows height classes
# channel index i (0-based) corresponds to class id i+1 in label_buffer
# --------------------------------------------------------------------------
CLASS_CODE = ["1a", "1b", "1c", "2", "3", "4", "6"]
CLASS_LO_M = [0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00]
# the top class is open-ended; 9.0 is a conservative working ceiling and is
# only used for the midpoint, never for a physical claim
CLASS_HI_M = [1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 9.00]
CLASS_MID_M = [(a + b) / 2.0 for a, b in zip(CLASS_LO_M, CLASS_HI_M)]
N_CLASSES = 7

# UKCEH validation against Countryside Survey field data
REF_EXACT_AGREEMENT = 0.32
REF_WITHIN_ONE_AGREEMENT = 0.63

# --------------------------------------------------------------------------
# PROVISIONAL: above-ground biomass per unit length, by height class.
# These are placeholders so the propagation machinery can be exercised.
# Replace with values from the hedgerow biomass literature and record the
# source for each row before publishing anything derived from them.
# Units: tonnes of above-ground dry biomass per km of hedge, (low, best, high)
# --------------------------------------------------------------------------
BIOMASS_T_PER_KM = {
    "1a": (2.0, 4.0, 7.0),
    "1b": (5.0, 9.0, 14.0),
    "1c": (9.0, 15.0, 23.0),
    "2":  (16.0, 26.0, 39.0),
    "3":  (28.0, 45.0, 66.0),
    "4":  (45.0, 72.0, 105.0),
    "6":  (70.0, 115.0, 170.0),
}
BIOMASS_IS_PROVISIONAL = True
CARBON_FRACTION = 0.475          # IPCC default carbon fraction of dry biomass
CARBON_FRACTION_SD = 0.02

# --------------------------------------------------------------------------
# experiment matrix, section 2 of the work plan
# --------------------------------------------------------------------------
VARIANTS = ["Simple", "SRA", "TTT"]
VARIANT_LABEL = {
    "Simple": "Concatenation",
    "SRA": "SR cross-attention",
    "TTT": "TTT cross-attention",
}
SEEDS = [42, 1337, 2024]
DATA_FRACTIONS = [0.25, 0.50, 1.00]
EPOCHS = 100
SELECT_CKPT_ON = "val_macro_relaxed_f1"   # one criterion for every run

# --------------------------------------------------------------------------
# paths, override with environment variables
# --------------------------------------------------------------------------
BASE_DIR = os.environ.get("HF_BASE_DIR", "./data")
OUT_DIR = os.environ.get("HF_OUT_DIR", "./refpack_out")
SPLIT_DIR = os.path.join(OUT_DIR, "splits")
RUN_DIR = os.path.join(OUT_DIR, "runs")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

for _d in (OUT_DIR, SPLIT_DIR, RUN_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)


def class_name(i: int) -> str:
    """0-based channel index to UKCEH class code."""
    return CLASS_CODE[i]


def class_range_str(i: int) -> str:
    if i == N_CLASSES - 1:
        return f">= {CLASS_LO_M[i]:.2f}"
    return f"{CLASS_LO_M[i]:.2f}-{CLASS_HI_M[i]:.2f}"

# --------------------------------------------------------------------------
# FAST TRACK, for the GeoCV 2027 deadline
#
# The logs show every variant reaching its best validation F1 by epoch 22 and
# its minimum validation loss by epoch 7. A 30-epoch cap with patience 8 on
# the selection metric therefore loses nothing that was actually measured,
# and it is the single largest saving available.
# --------------------------------------------------------------------------
FAST_EPOCH_CAP = 30
FAST_PATIENCE = 8
FAST_VAL_EVERY = 1

# measured training throughput on the RTX 4060 laptop, patches per second
THROUGHPUT = {"Simple": 9.30, "SRA": 9.09, "TTT": 5.59}
# single-encoder models carry roughly half the encoder cost. ESTIMATE, not
# measured; refine after the first epoch of the first single-encoder run.
THROUGHPUT_SINGLE_ENCODER = 16.0
# measured validation minutes per pass
VAL_MINUTES = {"Simple": 3.7, "SRA": 3.7, "TTT": 5.5, "single": 2.5}
FULL_TRAIN_PATCHES = 26_130
SMALL_TRAIN_PATCHES = 16_600      # the set the existing TTT run and early SRA used
# early stopping typically ends a run well before the cap
EARLY_STOP_SAVING = 0.25
