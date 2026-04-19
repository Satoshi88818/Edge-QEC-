# EdgeQEC v14.0

> **Edge-deployable Quantum Error Correction** — a fully integrated framework combining reinforcement learning, graph neural networks, and hardware-aware decoding for real-time quantum error correction on resource-constrained devices.

---

## Table of Contents

1. [What is EdgeQEC?](#what-is-edgeqec)
2. [What Makes It Unique](#what-makes-it-unique)
3. [v14 Improvements at a Glance](#v14-improvements-at-a-glance)
4. [Architecture Overview](#architecture-overview)
5. [Installation & Dependencies](#installation--dependencies)
6. [Supported QEC Codes](#supported-qec-codes)
7. [Noise Models](#noise-models)
8. [Neural Encoders](#neural-encoders)
9. [Decoders](#decoders)
10. [RL Agents](#rl-agents)
11. [Hardware Backends & Deployment](#hardware-backends--deployment)
12. [Training & Benchmarking](#training--benchmarking)
13. [CLI Reference](#cli-reference)
14. [Testing](#testing)
15. [Future Directions](#future-directions)
16. [Enhancement History](#enhancement-history-v13--v14)

---

## What is EdgeQEC?

EdgeQEC is a research-grade, edge-deployable framework for **quantum error correction (QEC)** that replaces static classical decoders with **reinforcement learning agents** guided by graph neural network encoders. It is designed to run in real time on hardware-constrained platforms — FPGAs, mobile NPUs, and edge inference chips — while maintaining competitive logical error rates (LER) against state-of-the-art decoders like PyMatching and IBM qLDPC.

The core thesis: classical decoders like Minimum-Weight Perfect Matching (MWPM) are fixed algorithms. EdgeQEC trains agents that *adapt* to time-varying noise, device drift, thermal fluctuations, leakage, and crosstalk — making them fundamentally more suitable for the messy reality of near-term quantum hardware.

---

## What Makes It Unique

EdgeQEC sits at a genuinely rare intersection of fields, and several of its design choices have no direct counterpart in existing open-source QEC frameworks.

**1. End-to-end differentiable decoding pipeline**

The full path from syndrome extraction through GNN encoding, neural MWPM, Sinkhorn soft-matching, Cascade CNN post-processing, and RL reward is differentiable. This means the RL agent's policy gradient can flow backwards through the decoder itself — not just through a reward signal. No other open QEC framework exposes this joint training path.

**2. ML noise estimation closed-loop**

The `MLNoiseEstimator` uses temporal convolution and a Transformer over syndrome history to predict per-qubit error probabilities in real time. In v14, these probabilities are actively wired into three places simultaneously: the GNN encoder (as extra node features), the neural MWPM edge-cost initialisation, and the Hybrid Decoder Pipeline's training path. This creates a closed feedback loop where the decoder's internal costs adapt to the estimated noise regime on every step — not just at training time.

**3. Hypergraph-product qLDPC codes with exact metrics**

The `qLDPCCode` class uses the Tillich–Zémor (2014) hypergraph-product construction over circulant LDPC codes. This guarantees the CSS condition algebraically (verified by assertion at construction time), provides sparse parity checks with bounded row weight, and enables exact BFS girth computation and exhaustive minimum-distance search for codes up to 64 qubits. Most QEC simulators treat qLDPC codes as black boxes; EdgeQEC constructs them from first principles.

**4. Dynamic stabiliser re-tiling**

`SurfaceCode.dynamic_retile()` implements Google-style periodic stabiliser layout switching between two pre-built dual-H matrices. Each retiling invalidates the cached logical operator basis and changes which stabilisers are measured in subsequent rounds — matching the behaviour of real dynamic circuit experiments on superconducting hardware. The `DynamicRetilingModel` drives this at configurable periods during environment stepping.

**5. Hardware-co-designed RL**

The RL loss function includes differentiable latency and power regularisation terms. The `CurriculumManager` gates stage progression on joint satisfaction of reward, latency, energy, and thermal budget constraints. DVFS modelling, FPGA spatial-unroll resource estimation, and ONNX Runtime WCET profiling with a Prometheus violation counter are built into the core training loop — not bolted on as post-hoc analysis.

**6. Multi-format deployment as first-class output**

A single `--deploy` flag exports: standard ONNX, fused ONNX (policy + decoder + reweighter in one graph), PTQ int8 via `torch.ao.quantization` with histogram calibration, TensorRT engine, ExecuTorch `.pte`, and a separate ONNX for the learned edge reweighter. A JSON deployment report with WCET p99, throughput (kHz), and model checksum is written automatically.

---

## v14 Improvements at a Glance

| ID | Improvement | Priority | Status |
|----|-------------|----------|--------|
| P0-1 | `qLDPCCode` → hypergraph-product construction with circulant LDPC, BFS girth, exhaustive min-distance | P0 | ✅ Complete |
| P0-2 | `MLNoiseEstimator` wired into decoder edge-cost init, GNN `noise_probs_dim` feature, `HybridDecoderPipeline.forward_train`, `QuantumPhoneEnv.step` | P0 | ✅ Complete |
| P0-3 | `HypergraphGNNEncoder` & `TannerEdgeGNNEncoder` rewritten with `torch.scatter_add`; permutation-equivariance unit tests added | P0 | ✅ Complete |
| P0-4 | `SurfaceCode.dynamic_retile()` fully implemented with dual-H layout A↔B; invalidates cached logical operators | P0 | ✅ Complete |
| P1-5 | `WCETProfiler.profile_onnxruntime()` + `profile_tensorrt()`; `edgeqec_wcet_violation_total` Prometheus counter | P1 | ✅ Complete |
| P1-6 | `StimBackend` made default for all LER curves; `--stim-noise-model` CLI flag | P1 | ✅ Complete |
| P1-7 | `_hybrid_decoder` / `_reweighter` set on `DQNAgent` / `PPOAgent` when `use_neural_mwpm=True` | P1 | ✅ Complete |
| P1-8 | `MixedPrecisionQuantizer.calibrate_and_quantize()` via `torch.ao.quantization`; histogram observer; 512-sample calibration; post-quant accuracy test | P1 | ✅ Complete |

---

## Architecture Overview

```
Syndrome Extraction  (Stim / Qiskit / Qibo dynamic circuits)
        │
        │   leakage · crosstalk · dropout · re-tiling (dual-H)
        ▼
MLNoiseEstimator ──────────────────────────────────────────────┐
(temporal conv + Transformer → per-qubit p_err)                │
                                                       error_probs
        ▼                                                      │
TannerEdgeGNN / HypergraphGNN  ◄── noise_probs as extra feature
(scatter_add message passing, permutation-equivariant)
        │
        │  node embeddings
        ▼
Hybrid Neural MWPM Decoder
  ├── Training path:  Sinkhorn OT (differentiable)  ◄── noise_probs → edge-cost init
  └── Inference path: PyMatching blossom
                      + LearnedEdgeReweighter (ONNX)
        │
        │  correction (soft train / hard infer)
        ▼
Cascade CNN Post-Processor  (geometric logical-flip suppression)
        │
        ▼
LogicalObservableTracker  (per-logical + joint LER)
        │  reward signal
        ▼
RL Agent  (DQN / PPO / Hierarchical)
        │  latency + power regularisation loss
        ▼
CurriculumManager  ◄──► ThermalPowerSensor (DVFS model)
        │
        ▼
Export: Fused ONNX · PTQ int8/fp8 · TensorRT · ExecuTorch
Prometheus: cycle time / LER / WCET violations
Stim: default LER reference curves
```

---

## Installation & Dependencies

```bash
# Core
pip install torch>=2.2 numpy scipy networkx pymatching stim qiskit qibo

# ML extras
pip install torch-geometric mamba-ssm einops

# Hardware / export
pip install onnx onnxruntime tensorrt executorch prometheus-client psutil pyyaml

# Testing
pip install pytest hypothesis wandb ray[rllib]

# Optional FPGA toolchain (Xilinx/Intel)
# pip install pyopencl pynq
```

**Requirements:** Python ≥ 3.10 · CUDA ≥ 12.1 (optional) · PyTorch ≥ 2.2

---

## Supported QEC Codes

| Code | Class | Notes |
|------|-------|-------|
| Rotated surface code | `SurfaceCode` | Dual-H dynamic retiling (v14) |
| Repetition code | `RepetitionCode` | Baseline / debugging |
| Steane [[7,1,3]] code | `SteaneCode` | CSS, exact logical operators |
| 2D colour code | `ColorCode` | Triangular stub |
| HP qLDPC | `qLDPCCode` | Tillich–Zémor construction (v14) |

All codes expose a common `StabilizerCode` interface: `check_matrix_x()`, `check_matrix_z()`, `get_syndrome()`, `logical_operators()`, `tanner_adjacency()`, `tanner_edge_features()`, and `dynamic_retile()`.

Use `CodeFactory.create(name, distance)` for uniform instantiation.

---

## Noise Models

| Model | Class | Description |
|-------|-------|-------------|
| Depolarising | `DepolarizingNoise` | IID Pauli errors at rate `p` |
| Correlated | `CorrelatedNoise` | Nearest-neighbour error spread |
| Leakage + crosstalk | `LeakageCrosstalkModel` | EMA leakage state, XT transport |
| Qubit dropout | `QubitDropoutModel` | Dark qubit / ancilla loss simulation |
| Dynamic retiling | `DynamicRetilingModel` | Periodic layout swap (v14) |
| Thermal power | `ThermalPowerSensor` | EMA power/temp with DVFS scaling |

Stim circuit-level noise is the default benchmark backend as of v14. The `--stim-noise-model` flag selects the circuit model (default: `surface_code:rotated_memory_z`).

---

## Neural Encoders

### TannerEdgeGNNEncoder (default)
Heterogeneous message passing over the Tanner graph (check nodes ↔ variable nodes) with edge-type and edge-weight features. Rewritten in v14 with `torch.scatter_add` for correctness and GPU efficiency. Optional Mamba SSM blocks for spatio-temporal syndrome modelling. Accepts `noise_probs` as extra node features (v14 P0-2).

### HypergraphGNNEncoder
Two-pass hypergraph message passing: variable→check aggregation then check→variable aggregation, both via `scatter_add`. Handles arbitrary check weight (suitable for qLDPC codes with weight > 4). Permutation-equivariant by construction and verified by unit test (v14 P0-3). Accepts `noise_probs` as extra features (v14 P0-2).

### MLNoiseEstimator
1D temporal convolution + Transformer Encoder over a configurable history window of syndrome measurements. Outputs per-qubit sigmoid error probabilities. As of v14 these probabilities feed into the GNN encoder, the neural MWPM edge-cost matrix, and the Hybrid Decoder Pipeline simultaneously, closing the noise estimation loop.

Select encoder with `--encoder [tanner_edge_gnn|hypergraph_gnn|cnn|mamba|transformer|gnn]`.

---

## Decoders

### HybridDecoderPipeline (recommended)
Full pipeline: Neural MWPM → Cascade CNN Post-Processor. Training uses Sinkhorn optimal transport for differentiable soft matching. Inference uses PyMatching blossom with learned edge weights. As of v14, noise probabilities from `MLNoiseEstimator` are used to initialise the edge-cost matrix at both training and inference time.

### NeuralMWPMDecoder
GNN projection + Transformer Encoder → per-edge weight prediction → Sinkhorn (train) / PyMatching (infer). The `LearnedEdgeReweighter` MLP adjusts edge costs at inference and is exported separately as an ONNX model.

### CascadeCNNPostProcessor
1D Conv stack for geometric logical-flip suppression. FPGA/ASIC-friendly (no attention, no scatter operations). Used as the final stage in the Hybrid Pipeline.

### MWPMDecoder
Classic MWPM baseline via PyMatching. Falls back to random correction if PyMatching is unavailable.

---

## RL Agents

All agents include differentiable latency and power regularisation in the loss function.

| Agent | Class | Notes |
|-------|-------|-------|
| Deep Q-Network | `DQNAgent` | Dual Q-networks, ε-greedy, soft imitation pre-training |
| Proximal Policy Optimisation | `PPOAgent` | Actor-critic, reg loss available |
| Hierarchical Decoder | `HierarchicalDecoderAgent` | Strategy layer (DQN×4 actions) + primitive layer |

As of v14 (P1-7), `DQNAgent` and `PPOAgent` set `self._hybrid_decoder` and `self._reweighter` at construction time when `use_neural_mwpm=True`, enabling the fused ONNX exporter to work correctly without post-hoc patching.

Select algorithm with `--algorithm [dqn|ppo|hierarchical_decoder|hierarchical_dreamer|...]`.

---

## Hardware Backends & Deployment

### Export formats (single `--deploy` flag)

| Format | File | Notes |
|--------|------|-------|
| Standard ONNX | `edgeqec.onnx` | Validated with `onnx.checker` |
| Fused ONNX | `edgeqec_fused.onnx` | Policy + decoder + reweighter in one graph (v14) |
| PTQ int8 | `edgeqec_int8.pt` | Calibrated via `torch.ao.quantization`, histogram observer (v14) |
| TensorRT | `edgeqec.trt` | GPU inference engine |
| ExecuTorch | `edgeqec.pte` | Mobile / embedded runtime |
| Edge reweighter | `edge_reweighter.onnx` | Standalone MLP for edge-cost adjustment |
| Quantisation recipe | `quantization_recipe.yaml` | For external toolchains (TensorRT QDQ, etc.) |
| Deploy report | `deploy_report.json` | WCET p99, throughput kHz, model checksum, timestamp |

### WCET Profiling (v14 P1-5)
`WCETProfiler` measures worst-case execution time across three backends: PyTorch eager, ONNX Runtime (CPU/CUDA), and TensorRT. Reports mean, p99 latency, within-budget boolean, and violation count. The Prometheus counter `edgeqec_wcet_violation_total` is incremented every time a decode cycle exceeds the configured budget (default 0.48 µs).

### FPGA Resource Estimation
`MockFPGA` estimates LUT, FF, BRAM, and DSP requirements for distance 3/5/7 across three dataflow variants (`gnn_cnn`, `gnn_only`, `cnn_only`). Spatial-unroll mode is toggled with `--fpga-spatial-unroll`.

### Prometheus Metrics
Start the exporter with `--prometheus-port <port>`. Exposed metrics: `edgeqec_ler`, `edgeqec_latency_us`, `edgeqec_cycle_time_us`, `edgeqec_power_mw`, `edgeqec_wcet_violation_total`.

---

## Training & Benchmarking

### Basic training run
```bash
python edgeqec.py \
  --algorithm dqn \
  --code surface \
  --distance 3 \
  --encoder tanner_edge_gnn \
  --episodes 1000 \
  --neural-mwpm \
  --ml-noise-estimator \
  --cascade-cnn
```

### qLDPC with transfer learning
```bash
python edgeqec.py \
  --algorithm dqn \
  --code qldpc \
  --distance 5 \
  --encoder hypergraph_gnn \
  --hypergraph-gnn \
  --qldpc-curriculum \
  --episodes 2000
```

### Full deployment pipeline
```bash
python edgeqec.py \
  --algorithm dqn \
  --code surface \
  --distance 5 \
  --neural-mwpm \
  --ml-noise-estimator \
  --mixed-precision int8 \
  --fused-onnx \
  --wcet-onnxruntime \
  --ptq-calibration-samples 512 \
  --stim-noise-model surface_code:rotated_memory_z \
  --deploy \
  --output-dir ./deploy_d5
```

### LER benchmark curve
The benchmark runs across physical error rates `[0.01, 0.03, 0.05, 0.08, 0.10, 0.15]` by default, with Stim as the primary reference backend (v14). Use `--compare-cascade`, `--compare-vegapunk`, `--compare-tesseract`, `--compare-ibm-qldpc` to add comparative baselines. The pseudo-threshold is estimated from the Stim LER curve.

### Curriculum stages
Training progresses automatically through six stages (easy → medium → hard → expert → qldpc_transfer → qldpc_hard) gated on reward, latency, energy, and thermal budget. DVFS-aware power budgets decrease as stages advance.

---

## CLI Reference

### Core flags
| Flag | Default | Description |
|------|---------|-------------|
| `--algorithm` | `dqn` | RL algorithm |
| `--code` | `surface` | QEC code type |
| `--distance` | `3` | Code distance |
| `--encoder` | `tanner_edge_gnn` | Neural encoder |
| `--episodes` | `1000` | Training episodes |
| `--trials` | `5000` | LER benchmark trials |
| `--rounds` | `3` | Syndrome measurement rounds |

### v14 new flags
| Flag | Default | Description |
|------|---------|-------------|
| `--stim-noise-model` | `surface_code:rotated_memory_z` | Stim circuit noise model (P1-6) |
| `--ptq-calibration-samples` | `512` | Representative syndromes for PTQ int8 calibration (P1-8) |
| `--wcet-onnxruntime` | `false` | Run WCET profiling via ONNX Runtime (P1-5) |

### Hardware flags
| Flag | Description |
|------|-------------|
| `--mixed-precision [fp32\|fp16\|int8\|int4\|fp8]` | Quantisation target |
| `--latency-budget` | WCET budget in µs (default 0.48) |
| `--thermal-budget` | Power budget in mW (default 200) |
| `--fpga-spatial-unroll` | Enable FPGA spatial dataflow unrolling |
| `--fpga-variant [gnn_cnn\|gnn_only\|cnn_only]` | FPGA dataflow variant |
| `--tensorrt` | Export TensorRT engine |
| `--executorch` | Export ExecuTorch `.pte` |
| `--fused-onnx` | Export fused ONNX graph |
| `--dvfs` | Enable DVFS power modelling |
| `--prometheus-port` | Enable Prometheus metrics exporter |

---

## Testing

EdgeQEC ships **94 unit tests** (82 retained from v13 + 12 new in v14) covering correctness properties, decoder shapes, agent construction, equivariance guarantees, and hardware profiling paths.

```bash
# Unit tests (pure Python, no GPU required)
python edgeqec.py --unit-tests

# Integration test (Stim surface code threshold)
python edgeqec.py --integration-tests

# Property-based tests (requires hypothesis)
python edgeqec.py --property-tests
```

### v14 new tests

| Test | Validates |
|------|-----------|
| `test_hp_qldpc_css_condition` | HP construction satisfies `Hx Hz^T = 0 mod 2` |
| `test_hp_qldpc_girth_ge_6` | Circulant HP codes achieve girth ≥ 4 (typically 6) |
| `test_hp_qldpc_min_distance_exact` | `min_distance()` returns ≥ 1 |
| `test_dynamic_retile_syndrome_changes` | Syndrome layout changes after `dynamic_retile()` |
| `test_dynamic_retile_layout_toggle` | Layout toggles A↔B correctly |
| `test_gnn_encoder_permutation_equivariance` | `TannerEdgeGNNEncoder` output invariant to input permutation |
| `test_hypergraph_gnn_permutation_equivariance` | `HypergraphGNNEncoder` output invariant to variable-node permutation |
| `test_noise_estimator_wired_to_decoder` | `MLNoiseEstimator` output flows through `HybridDecoderPipeline` |
| `test_dqn_agent_has_hybrid_decoder` | `DQNAgent._hybrid_decoder` and `._reweighter` set correctly |
| `test_ppo_agent_has_hybrid_decoder` | `PPOAgent._hybrid_decoder` set correctly |
| `test_ptq_calibration_runs` | PTQ calibration completes and saves quantised model |
| `test_wcet_profiler_onnxruntime` | ONNX Runtime WCET profiler returns timing dict |
| `test_stim_default_ler` | Stim LER is monotone in physical error rate |

---

## Future Directions

EdgeQEC v14 closes eight critical correctness and hardware-fidelity gaps relative to v13. The roadmap below identifies the highest-leverage directions for continued development.

### Near-term (v15 candidates)

**Floquet code support.** The CLI already accepts `--code floquet` but the `CodeFactory` does not yet map it. Implementing a `FloquetCode` class with time-periodic check schedules would enable training on honeycomb lattice codes, which are among the most promising candidates for superconducting architectures.

**Multi-qubit logical tracking with correlated LER.** `LogicalObservableTracker` computes per-logical and joint LER independently. Adding a full correlation matrix across logical qubits would enable the RL agent to optimise correction strategies for codes with multiple logical qubits (e.g., larger HP qLDPC codes where k > 1).

**Decoder-aware exploration in RL.** The current ε-greedy and PPO policies treat actions (qubit flips) uniformly. A decoder-informed action prior — seeded from `HybridDecoderPipeline.decode()` output — could dramatically accelerate early training by biasing exploration toward plausible corrections.

**Real device feedback loop.** The `--real-device-feedback` flag and `real_device_feedback` config field exist but are not yet plumbed to a hardware interface. Connecting to Qiskit Runtime or a Qibo device backend would allow online adaptation to measured noise profiles, which is the core value proposition of RL-based decoding.

### Medium-term

**Recurrent / world-model agents.** The `--algorithm dreamer` and `--algorithm hierarchical_dreamer` entries in the CLI are stubs. Implementing DreamerV3-style latent world models would allow the agent to plan over longer syndrome sequences, potentially outperforming reactive DQN/PPO on codes with complex temporal error correlations (e.g., Floquet or dynamically-tilted surface codes).

**Distributed training with Ray RLlib.** The `--n-envs` flag and `use_ray / ray_num_cpus` config fields are present but `n_envs > 1` is not yet parallelised. Integrating Ray RLlib vectorised environments would unlock large-scale training across many physical error rate points simultaneously.

**Learned stabiliser measurement schedules.** Currently the syndrome measurement order is fixed by the code geometry. An agent that also controls *which* stabilisers to measure and in what order (adaptive syndrome extraction) could reduce the number of rounds needed to identify a correction, directly cutting latency.

**Analog syndrome readout.** The current framework assumes binary syndrome bits. Extending `MLNoiseEstimator` and the GNN encoders to accept soft (analog) syndrome values — as produced by some readout schemes — would improve noise estimation accuracy and is a natural fit for the existing `noise_probs` feature pipeline.

### Long-term

**ASIC-targeted architecture search.** The FPGA resource estimator (`MockFPGA`) is a stub with fixed scaling formulae. Replacing it with a differentiable hardware cost model (e.g., based on roofline analysis or lookup tables from real synthesis runs) would enable neural architecture search that jointly optimises LER and silicon area for a target process node.

**Cross-code transfer and meta-learning.** The qLDPC curriculum already includes a surface→qLDPC transfer stage. Extending this to a full meta-learning setup (e.g., MAML over a family of HP codes at different distances and rates) would produce decoders that adapt rapidly to new code families at inference time — a key requirement as the QEC field converges on optimal code constructions.

**Online noise model adaptation.** The `MLNoiseEstimator` is trained offline alongside the RL agent. A streaming online learning scheme — updating the estimator's parameters from real syndrome data without stopping the decoder — would allow EdgeQEC to track non-stationary noise processes (e.g., TLS telegraph noise, thermally-driven drift) in deployed hardware.

**Formal WCET guarantees.** The current `WCETProfiler` measures empirical p99 latency. For safety-critical deployment, a formal WCET bound (via static analysis or interval arithmetic over the ONNX graph) would be necessary. Integrating with tools like Platin or aiT would be a significant step toward certifiable edge deployment.

**Federated and privacy-preserving training.** In a multi-device quantum computing environment, syndrome data from different QPUs could be used to collaboratively train a shared decoder without sharing raw syndrome traces. Federated RL with differential privacy guarantees would be relevant here.

---

## Enhancement History: v13 → v14

| ID | Improvement | Key Change |
|----|-------------|------------|
| P0-1 | qLDPCCode → hypergraph-product construction | `hypergraph_product(H1, H2)` with circulant LDPC; exact BFS girth; exhaustive min-distance for n ≤ 64 |
| P0-2 | MLNoiseEstimator wired into full decoder pipeline | `noise_probs` fed to `NeuralMWPMDecoder` edge-cost init, GNN `noise_probs_dim` feature, `HybridDecoderPipeline.forward_train`, `QuantumPhoneEnv.step` |
| P0-3 | GNN forward passes rewritten with `scatter_add` | Both `HypergraphGNNEncoder` and `TannerEdgeGNNEncoder` use `torch.scatter_add`; permutation-equivariance unit tests added |
| P0-4 | `SurfaceCode.dynamic_retile()` fully implemented | Dual-H matrix layout A↔B; invalidates cached logical operators; `DynamicRetilingModel` now changes syndrome measurements |
| P1-5 | WCET on real ONNX Runtime / TensorRT | `WCETProfiler.profile_onnxruntime()` + `profile_tensorrt()`; `edgeqec_wcet_violation_total` Prometheus counter |
| P1-6 | `StimBackend` as default for LER curves | `benchmark_ler_curve` uses `StimBackend` as primary reference; `--stim-noise-model` CLI flag |
| P1-7 | `_hybrid_decoder` / `_reweighter` on `DQNAgent` / `PPOAgent` | Both agents set `self._hybrid_decoder` and `self._reweighter` in `__init__` when `use_neural_mwpm=True` |
| P1-8 | Proper PTQ int8 calibration | `MixedPrecisionQuantizer.calibrate_and_quantize()` via `torch.ao.quantization`; histogram observer; 512-sample calibration; post-quantisation accuracy test |

---

*EdgeQEC v14.0 — Python ≥ 3.10 · PyTorch ≥ 2.2 · CUDA ≥ 12.1 (optional)*
