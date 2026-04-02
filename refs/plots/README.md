# Presentation plots

Generated with `python scripts/generate_presentation_plots.py`.

These figures are split between:
- `data`: built from canonical values already present in `refs/` and the Beamer deck.
- `mixed`: a compact synthesis figure using current project status.
- `schematic`: explanatory figures intended for presentation framing rather than raw benchmark reporting.
- `proxy`: layer curves that are not official fastabx but help bridge to Vadim/Bruny depth analyses.

## Files

### `01_monolingual_frontend_comparison.png`
- Title: Monolingual eng1 10 min frontend comparison
- Type: data
- Description: Grouped CER/WER bars for HuBERT, minimal JEPA, WavJEPA HF, and local WavJEPA checkpoint.
- Also exported as: `01_monolingual_frontend_comparison.pdf`

### `02_delta_vs_hubert.png`
- Title: Delta vs HuBERT baseline
- Type: data
- Description: Horizontal delta bars showing how far each JEPA/WavJEPA option sits from HuBERT on CER and WER.
- Also exported as: `02_delta_vs_hubert.pdf`

### `03_abx_vs_asr_scatter.png`
- Title: ABX vs ASR scatter
- Type: data
- Description: Two-panel scatter showing that HuBERT and WavJEPA HF differ on ABX while tying on CER/WER in the current eng1 10 min slice.
- Also exported as: `03_abx_vs_asr_scatter.pdf`

### `04_multilingual_scaling.png`
- Title: Multilingual ASR scaling 10 min to 1 h
- Type: data
- Description: Slope chart for multilingual ASR-only showing improved CER and WER when moving from 10 min to 1 h.
- Also exported as: `04_multilingual_scaling.pdf`

### `05_multilingual_tradeoffs.png`
- Title: Multilingual trade-offs
- Type: data
- Description: Grouped bars comparing multilingual ASR-only, ASR+LID, and LoRA settings.
- Also exported as: `05_multilingual_tradeoffs.pdf`

### `06_runtime_comparison.png`
- Title: Measured runtime comparison
- Type: data
- Description: Horizontal bar chart of representative L4 training times from the runtime table.
- Also exported as: `06_runtime_comparison.pdf`

### `07_performance_vs_cost.png`
- Title: Performance vs cost
- Type: data
- Description: Scatter of representative runs with runtime on x-axis and WER on y-axis.
- Also exported as: `07_performance_vs_cost.pdf`

### `08_experiment_coverage_heatmap.png`
- Title: Experiment coverage heatmap
- Type: mixed
- Description: Heatmap summarizing what is done, partial, or planned across model families and evaluation axes.
- Also exported as: `08_experiment_coverage_heatmap.pdf`

### `09_multilingual_queue_timeline.png`
- Title: Multilingual queue timeline
- Type: data
- Description: Gantt-like timeline of completed multilingual queue stages from the runtime log.
- Also exported as: `09_multilingual_queue_timeline.pdf`

### `10_scope_publishable_roadmap.png`
- Title: Scope vs publishable roadmap
- Type: schematic
- Description: Three-column roadmap separating delivered content, strong next steps, and optional extensions.
- Also exported as: `10_scope_publishable_roadmap.pdf`

### `11_hf_local_drift_summary.png`
- Title: HF vs local drift summary
- Type: data
- Description: Compact drift summary from the 311-utterance analysis slide.
- Also exported as: `11_hf_local_drift_summary.pdf`

### `12_protocol_fidelity_vs_speedup.png`
- Title: Protocol fidelity vs speedup
- Type: schematic
- Description: Schematic scatter for the speech_encoder discussion: strict protocol fidelity is not where the largest speedups usually live.
- Also exported as: `12_protocol_fidelity_vs_speedup.pdf`

### `13_asr_vs_ser_universality.png`
- Title: ASR vs SER universality map
- Type: data
- Description: Scatter combining Janis monolingual WER with Bruny RAVDESS UAR (full regime).
- Also exported as: `13_asr_vs_ser_universality.pdf`

### `14_ranking_inversion_across_tasks.png`
- Title: Ranking inversion across tasks
- Type: data
- Description: Rank heatmap: ASR uses inverse WER; ABX uses inverse error from the deck; SER from report.tex appendix.
- Also exported as: `14_ranking_inversion_across_tasks.pdf`

### `15_layer_phonetic_proxy.png`
- Title: Layer-wise phonetic proxy
- Type: proxy
- Description: Curve from layer_phonetic_proxy.py on exp/abx_layers/*_full (complements Vadim CTC weights + Bruny SER depth).
- Also exported as: `15_layer_phonetic_proxy.pdf`

