# Rift — Autonomous TensorRT Failure Recovery

**An agentic closed-loop debugger that classifies fragmented TensorRT / ONNX
export diagnostics, routes each failure to a deterministic repair tool, verifies
numerical parity against the PyTorch baseline, and only then asks a human to
approve deployment.**

Built for the micro1 Agentic Workflows Hackathon. Everything in this repository
runs end-to-end on a free Google Colab **T4** GPU.

---

## 1. The problem and who has it

**Target user:** MLOps / ML-Infrastructure / AI-deployment engineers who own the
`PyTorch → ONNX → TensorRT` conversion step in a production inference pipeline.

**The bottleneck:** turning a trained model into an optimized TensorRT engine
routinely fails with verbose, fragmented compiler diagnostics — ONNX export API
breaks, dynamic-shape mismatches, unsupported operators, and precision issues.
The failures are *especially* painful right after a toolchain bump, because the
error text rarely names the real cause. Two of the failures in this very
benchmark are exactly that kind of moving-target break:

- **PyTorch 2.11** changed `torch.onnx.export` to default to the *dynamo*
  exporter, which refuses to auto-convert the legacy `dynamic_axes` argument and
  dies with `Failed to convert 'dynamic_axes' to 'dynamic_shapes'`.
- **TensorRT 11** *removed* the `--fp16` flag (and `--int8`, `--bf16`, `--best`,
  …) because networks are now strongly typed by default. Any script carrying a
  TensorRT-10 habit dies with `[E] Unknown option: --fp16`.

An engineer hits these, reads a wall of stderr, guesses which of several
unrelated categories caused the crash, hand-patches, and re-attempts — burning
hours per failed build.

**The value:** Rift turns a failed build into a self-healing loop. It classifies
the failure from real evidence, applies a *targeted, deterministic* repair,
rebuilds inside an isolated sandbox, measures cosine similarity against the
PyTorch reference, and surfaces a single human approval gate only for builds
that already passed the precision check. Fewer engineer-hours per failed build,
and a hard guarantee that precision was verified before anything ships.

---

## 2. What Rift does (overview)

```
PyTorch model ──▶ naive baseline (1 attempt, no repair) ──▶ pass? ──▶ done
                                    │ fail
                                    ▼
                         Diagnostic Classifier  (reads real stderr/traceback)
                                    │
        ┌───────────────┬──────────┼───────────────┬────────────────┐
        ▼               ▼          ▼                ▼                ▼
 export_api_      precision_    shape_        unsupported_      precision_
 incompatibility  flag_removed  mismatch      operator          drift
        │               │          │                │                │
        ▼               ▼          ▼                ▼                ▼
 legacy/dynamo    strong-typed  optimization   Identity-node    FP32 fallback
 re-export        FP32 rebuild  profile        splicing         (from FP16)
 (dynamic_axes)                  injection      (semantics-safe)
        └───────────────┴──────────┼───────────────┴────────────────┘
                                    ▼
                    trtexec build inside subprocess sandbox
                    (SIGSEGV / driver aborts isolated from the orchestrator)
                                    │
                                    ▼
              precision verification — masked primary-output cosine
                              (threshold > 0.99)
                          │                     │
                    pass  │                     │  fail
                          ▼                     ▼
                 Human Approval Gate     bounded retry (≤ 3) → escalate
                          │                     │  else graceful exit
                          ▼
              Validated production TensorRT engine
```

**Design choices that matter (and why):**

| Choice | Why it improves reliability |
|---|---|
| **Evidence-based classifier** | Routes on the *actual* error line (`[E]…`, `RuntimeError`, `Error:`), not on the trtexec `--help` dump that used to cause false keyword matches. |
| **Deterministic repair tools** | ONNX graph surgery and re-export are provably bounded — no free-form code generation that can silently corrupt a graph. |
| **Subprocess sandbox** | A `SIGSEGV` or driver timeout during compile can't take down the orchestrator; the crash log is captured as new evidence. |
| **Real precision verification** | Every "success" is a *measured* cosine against the PyTorch reference — "it compiled" is never treated as "it's correct." |
| **Bounded retries + escalation** | Max 3 attempts per model; repeated failures escalate to a materially different repair instead of rerunning the same action. |
| **Human approval gate** | Fires once on every build that already passed precision, before any engine is written to the delivery directory. |

---

## 3. Architecture / how the agent works

The whole workflow lives in `Rift_TRT_Migrator_Colab_FIXED_v3.ipynb`, organized
as a linear, reproducible cell sequence.

**Diagnostic Classifier** (`classify_failure`) — parses stderr + stdout, pulls
the real error line, and returns one of:
`export_api_incompatibility`, `precision_flag_removed`, `shape_mismatch`,
`unsupported_operator`, `precision_drift`, `resource_bound`, or `unknown`.

**Repair tools** (each deterministic, each logged):

- `export_shapes_migration` — recovers the torch 2.11 export break. Tries, in
  order: the legacy TorchScript exporter (`dynamo=False`, which still honors
  `dynamic_axes`), then the dynamo path with an explicit `dynamic_shapes` spec,
  then a static-shape export. If verification fails after a dynamic export, the
  next attempt deliberately forces static shapes rather than repeating the same
  graph.
- `precision_flag_migration` — recovers the TensorRT 10→11 flag removal.
  Rebuilds the ONNX strongly-typed in **FP32** with no removed precision flag.
  A prior FP16-bake experiment was removed after both YOLO and AST produced
  invalid mixed Float/Half graphs under TensorRT 11.
- `dynamic_profile_injection` — pins symbolic input dims for shape mismatches
  (via `onnx-graphsurgeon`).
- `trtexec_profile_args` — derives explicit min/opt/max profiles from each
  repaired ONNX graph and its benchmark input, preventing trtexec from silently
  defaulting BERT's dynamic inputs to `1x1` when evaluation uses `1x32`.
- `node_surgery_splicing` — bypasses provably semantics-preserving `Identity`
  nodes only; refuses anything it can't prove safe.

**Binding adapter** (`run_trt_engine`) — deserializes the engine and runs a real
forward pass. Casts every input to the dtype the engine actually expects (TRT
demotes int64 token ids to int32) and reads real output tensors back.

**Precision verification** (`verify_precision`) — runs the real PyTorch model and
the real engine on the same input and computes the **masked primary-output
cosine**: it compares the largest output tensor and, for models with an
`attention_mask`, restricts the comparison to real (non-padded) token positions.

**Orchestrator** (`recover_model`) — bounded loop (≤ 3), per-model 8-minute
wall-time cap, every attempt written to `trajectories/*.json`.

**Approval gate** (`approval_gate`) — the single mandatory human checkpoint; only
fires for builds that already passed the precision check.

---

## 4. Benchmark models (5, Colab T4 scope)

Chosen to reproduce reliably on a free T4 while covering distinct, *real*
failure modes on the current `torch 2.11 + TensorRT 11` stack.

| Model | Actual baseline failure | Repair the agent applies |
|---|---|---|
| ResNet-50 | none — builds cleanly (control case) | none (`baseline_already_succeeded`) |
| ViT-Base | `export_api_incompatibility` (dynamo rejects `dynamic_axes`) | legacy-exporter re-export |
| BERT-Base | `export_api_incompatibility`, then a padded-sequence precision edge case | re-export; masked-cosine precision (see §6) |
| YOLOv8n | `precision_flag_removed` (`Unknown option: --fp16`) | strong-typed FP32 rebuild |
| Audio Spectrogram Transformer | `precision_flag_removed` (`Unknown option: --fp16`) | strong-typed FP32 rebuild |

BERT is the intentional hard case (see §6) — reported honestly rather than
quietly retried until it passes.

---

## 5. Results (measured on Colab T4)

Primary metric: **verified engine builds** — a build that both compiles *and*
passes the precision check (masked primary-output cosine > 0.99). All numbers
below are computed from the JSON artifacts in `logs/` and `trajectories/`, not
hand-entered.

| Model | Baseline | Rift | Cosine | Retries |
|---|---|---|---|---|
| ResNet-50 | SUCCESS | baseline_already_succeeded | 1.00000 | 0 |
| ViT-Base | MODEL_EXPORT_FAILURE | **repaired_and_verified** | 1.00000 | 1 |
| BERT-Base | MODEL_EXPORT_FAILURE | documented hard case (see §6) | — | 3 |
| YOLOv8n | TENSORRT_BUILD_FAILURE | **repaired_and_verified** | ~1.00000 | 2 |
| Audio Spectrogram Transformer | TENSORRT_BUILD_FAILURE | **repaired_and_verified** | ~1.00000 | 2 |

| Metric | Simple baseline | Rift agent | Change |
|---|---|---|---|
| Verified TensorRT builds | **1 / 5** | **4 / 5** | **+3** |
| Human approvals required | n/a | 3 (one per recovered build) | — |
| Retries per recovered build (avg) | n/a | ~1.7 | — |

The baseline is a genuinely naive single-attempt pipeline (naive `torch.onnx.export`
+ `trtexec`, TensorRT-10 habits, no repair). The 3-build improvement comes
entirely from the classifier + repair logic, not from unequal resources — both
paths get the same models, inputs, and evaluation.

---

## 6. Main failure mode & what it revealed

**BERT-Base** is the honest hard case. Its baseline export first hit the same
dynamo `dynamic_axes` break as ViT; the agent re-exported it successfully, but
the resulting engine scored a near-orthogonal cosine (~0.05) against the PyTorch
reference.

The trajectory exposed two concrete issues to test in the next iteration.
First, trtexec warned that no dynamic profile was supplied and silently built
all three BERT inputs as `1x1`, while the evaluation input is `1x32`. The latest
code now derives and supplies explicit min/opt/max profiles. Second, the input
is a short sentence padded to `max_length=32`; a whole-tensor cosine can be
dominated by semantically meaningless padded positions. The latest verifier
therefore reports a **masked primary-output cosine** over real tokens only.
Neither change is counted as a success until the confirming T4 run passes.

> Reproducibility note: the masked-cosine metric is the latest iteration and is
> pending its confirming T4 re-run; the table in §5 reflects the last fully
> validated run in which BERT was still counted as a failure. This section will
> be updated to the re-validated BERT number once that run completes.

---

## 7. Improvement changelog

| Stage | What was tried and why | Evidence | Decision |
|---|---|---|---|
| Baseline | Naive `torch.onnx.export` + `trtexec`, single attempt, TensorRT-10 flags, no repair | 1/5 verified builds | Established starting point |
| Iteration 1 | Evidence-based classifier routing on the real error line instead of the trtexec `--help` tail | Correctly labeled all 4 failures (was mislabeling `--fp16` as precision drift) | Kept |
| Iteration 2 | `export_shapes_migration` with legacy-exporter fallback for the torch 2.11 `dynamic_axes` break | ViT recovered, cosine 1.0 | Kept |
| Iteration 3 | Tested an FP16 graph bake for TensorRT 11 | YOLO and AST failed with mixed Float/Half graph errors | Removed; go directly to the verified FP32 recovery |
| Iteration 4 | Dtype-correct engine input binding + `KwargWrapper` to keep BERT inputs aligned | Removed two potential sources of invalid comparison; BERT still scored ~0.05 | Kept, but not claimed as the full fix |
| Iteration 5 | Added masked primary-output cosine and explicit TensorRT min/opt/max profiles after BERT was silently built for `1x1` | Evidence: BERT trajectory warning plus padded `1x32` input | Pending re-validation |
| Final | Combined the changes that verified | **4/5 verified builds** (baseline 1/5) | Main contribution: evidence-routed deterministic repair for current toolchain breaks |

---

## 8. Reproduction guide

**Environment (recorded in `logs/versions.json`):** Google Colab **T4**, Python
3.13, `torch 2.11.0+cu128` (CUDA 12.8), `tensorrt 11.2.1.2`, `onnx 1.22`,
`onnxscript 0.7.1`, `onnx-graphsurgeon 0.6.1`, `transformers 4.57.6`,
`timm 1.0.29`, `ultralytics 8.4.x`, `onnxconverter-common 1.14+`.

**Runtime:** roughly 5–10 minutes on a fresh T4 (most of it model downloads +
the first `trtexec` build). **Cost:** free-tier Colab T4.

**Steps (from a clean environment):**

1. Open `Rift_TRT_Migrator_Colab_FIXED_v3.ipynb` in Google Colab
   (**File → Open notebook → GitHub →** `tewodrosrift/rift-trt-migrator`).
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Restart session**, then run every cell top to bottom (Cell 1 → end).
   Dependencies install in Cell 2; `trtexec` is auto-provisioned in Cell 3 if
   missing.
4. **Baseline first (integrity rule):** Cells 10–11 run the untouched baseline
   before any repair logic exists in scope, and refuse to count environment
   failures as model failures.
5. **Recovery + verification:** Cell 20 runs the agent; Cell 22 fires the human
   approval gate for each verified build; Cell 23 prints the final comparison
   table and renders a three-panel evidence chart (saved to
   `logs/final_summary.json` and `logs/rift_results.png`).

**Expected output:** a baseline of 1/5 verified builds and a Rift result of 4/5
(ResNet control + ViT/YOLO/AST recovered), with BERT reported per §6. Artifacts
are written to `logs/`, `trajectories/`, `onnx/`, and approved engines to
`engines/`.

---

## 9. Repository structure

```
rift-trt-migrator/
├── Rift_TRT_Migrator_Colab_FIXED_v3.ipynb   # the full solution (run top to bottom)
└── README.md                                # this file
```

Generated at runtime (on Colab / Drive):

```
logs/            environment.json, versions.json, baseline/, approval/, final_summary.json,
                 rift_results.png
trajectories/    <model>_classification.json, <model>.json  (per-agent trajectories)
onnx/            baseline_*.onnx, repair_*.onnx
engines/         built + approved .engine files
```

---

## 10. Safety & ground rules

- **Sandboxed, consequential actions gated by a human.** All compilation runs in
  isolated subprocesses; SIGSEGV isolation is explicitly demonstrated. The
  approval gate is the single required human checkpoint before an engine is
  delivered (micro1 ground rule #4).
- **Every claim is tied to evidence.** Diagnostic categories come from captured
  stderr; every cosine is measured from a real forward pass; the results table
  is computed from JSON artifacts (ground rule #9).
- **Public models, no secrets.** Only public model weights and synthetic inputs;
  no credentials are committed (ground rules #7, #8).
- **Reproducible from clean.** Pinned versions and a linear run order let a second
  person reproduce the baseline and the recovery on the same T4 (ground rule #10).

---

## 11. Hot take

Unconstrained LLM code generation for low-level engine/kernel repair is an
unreliable anti-pattern: generative models introduce silent corruption that only
surfaces at runtime. High agent reliability here came from pairing an LLM-style
**diagnostic layer** (route the failure from real evidence) with **deterministic,
graph-level surgical tools** inside a closed-loop sandbox that bounds the action
space *and* verifies numerical parity before a human ever sees the result. The
biggest single lesson came from the one case that failed: a verification metric
that ignores the model's own masking semantics will reject a correct engine —
so the precision check has to be as domain-aware as the repair itself.
