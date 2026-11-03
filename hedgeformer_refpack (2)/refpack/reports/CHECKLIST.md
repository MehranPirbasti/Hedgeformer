# HedgeFormer, reference-grade checklist

## Phase 1, blocking

- [ ] **Lock the data facts**, section 1 **blocks submission**
      Provenance of the elevation layer, acquisition dates, co-registration and class support. The physical argument in the introduction is unsupported until the dates exist in the paper.
      entry point: `refpack.s1_data_audit`

- [ ] **Rerun all variants under one protocol**, section 2.1/2.3 (9 training runs) **blocks submission**
      Equal epochs, one checkpoint criterion, three seeds. The current margins sit in the fourth decimal and cannot be defended from a single seed.
      entry point: `refpack.s2_protocol.run_matrix`

- [ ] **Data-efficiency ablation**, section 2.2 (18 training runs)
      Turns the unequal data volume from a flaw into the strongest claim available: the adaptive operator leads by more when data is scarce.
      entry point: `refpack.s2_protocol.build_stratified_subsets`

- [ ] **External baselines**, section 3 (12 training runs) **blocks submission**
      Single-modality, early fusion, late fusion and a classical non-learning method. Without these there is no claim of state of the art, only an internal comparison.
      entry point: `refpack.s3_baselines`

- [ ] **Loss ablation**, section 4 (4 training runs) **blocks submission**
      Four loss components are claimed as contributions and none is ablated.
      entry point: `refpack.s4_ablation.ABLATION_GRID`

- [ ] **Parameters and inference cost**, section 2.4 **blocks submission**
      Three FILL entries in the complexity table, plus the seconds per square kilometre that the national-monitoring motivation implies.
      entry point: `refpack.s2_protocol.parameter_report`

- [ ] **Ordinal metrics and confusion matrix**, section 6.1 **blocks submission**
      Height error in metres is the number that makes the results legible next to the canopy-height literature and defuses the objection to a macro F1 of 0.43.
      entry point: `refpack.s6_metrics.ordinal_metrics`

- [ ] **Report the instance match rate**, section 9 **blocks submission**
      Matched-instance geometry improves trivially for a model that detects fewer and easier instances. The match rate is already computed as connectivity correctness; it only has to be reported next to every distance metric.
      entry point: `validation.FullMetrics`

## Phase 2, reference grade

- [ ] **Modality ablation at inference**, section 5.2
      Proves the mechanism directly and costs no training. The predicted cross-over between class 1a and class 6 is the result to look for.
      entry point: `refpack.s5_mechanism.modality_ablation`

- [ ] **Fast-weight probe**, section 5.1
      The signature figure. Fast-weight movement against canopy signal to noise, contrasted with the frozen baseline.
      entry point: `refpack.s5_mechanism.run_fast_weight_probe`

- [ ] **Synthetic phenology stress**, section 5.3
      Measures the leaf-off to leaf-on mismatch that motivates the whole architecture instead of asserting it.
      entry point: `refpack.s5_mechanism.phenology_curve`

- [ ] **Achievable ceiling**, section 6.3
      A general tool other groups will borrow, and the permanent answer to reviewers who object to absolute scores.
      entry point: `refpack.s6_metrics.achievable_ceiling`

- [ ] **Threshold-free length estimator**, section 7.1
      Length straight from the proximity field, with no threshold and no skeletonisation.
      entry point: `refpack.s7_length.length_from_field`

## Phase 3, lasting

- [ ] **True network length**, section 7.3
      The first defensible estimate of how much hedgerow actually exists in this landscape, and what share the reference product sees.
      entry point: `refpack.s7_length.network_length_report`

- [ ] **Landscape connectivity retention**, section 6.5
      Connects segmentation quality to ecological function.
      entry point: `refpack.s6_metrics.connectivity_retention`

- [ ] **Carbon bias of a binary map**, section 6.6
      The sentence that earns citations outside computer vision. Needs sourced biomass coefficients first.
      entry point: `refpack.s6_metrics.binary_map_carbon_bias`

- [ ] **Feature-level benchmark release**, section 8
      Publishable under the Airbus licence and, as far as we can tell, without precedent in this field.
      entry point: `refpack.s8_benchmark.export_features`

## Budget

- full programme: 43 training runs
- minimal defensible programme: 34 runs
- highest return if time is short: modality_dropout, ordinal, ceiling
