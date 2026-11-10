# scripts/refpack

The reference package that re-derives the manuscript numbers from the files under `results/`: data audit (s1), protocol (s2), baselines (s3), ablation (s4), mechanism and probe (s5), metrics (s6), network length (s7), benchmark (s8) and a driver (s9).

`config.py` carries shared constants. Two of them were provisional when this package was written and differ from the preparation script that produced the tiles: `DIST_CAP_M` and `BUFFER_M` are 5 m in the data (and in the paper), not 10 m. The constants are kept as found so that the package stays identical to the one used for the audit; the manuscript values were taken from the run configurations and from the preparation script.
