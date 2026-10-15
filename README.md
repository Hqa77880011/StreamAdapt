# StreamAdapt

Implementation of **StreamAdapt: Evidence-Gated Test-Time Adaptation for Reliable Model-State Transitions in Nonstationary Video Streams**.

Every live prediction comes from the committed model. A separate copy adapts on a past frame window, freezes, and is evaluated on later frames. A likelihood-ratio gate commits or rejects it; a retained predecessor supports rollback after degradation.

## Install

Use Python 3.11 or later in an activated environment:

```bash
python -m pip install -e .
```

CPU execution is the default. Install the appropriate CUDA PyTorch build and add `--device cuda` for GPU execution.

## Prepare video data

Use licensed Mini Kinetics-C, Mini SSV2-C, or UCF101 videos with a fixed corruption/order protocol. The repository accepts decoded-video files rather than assuming one release's directory structure. Retain official class IDs and group splits, then map class IDs to consecutive integers from zero.

Create a CSV manifest:

```csv
path,label,role,group,stream,order,boundary
raw/train_clip.avi,0,train,train_001,main,0,0
raw/val_clip.avi,0,val,val_001,main,0,0
raw/test_clip.avi,0,stream,test_001,main,0,1
```

Supply all three roles and all classes. A physical video group cannot cross roles. `order` defines clip order within a stream. `boundary=1` marks a predefined corruption/change boundary for recovery reporting. Videos with different action labels remain separate clips; temporal tracking and the frame window restart at clip cuts, while deployed model state and pending decisions persist across clips in the same stream.

```bash
streamadapt prepare --manifest data/videos.csv --output data/decoded --size 128 --stride 1
streamadapt train --data data/decoded --output outputs/source --epochs 50 --frames 8 --seed 13
```

Preparation decodes RGB frames, resizes them to a fixed square and preserves their order. `--stride` selects every nth original frame; reported delays/recovery use processed frames. Source training samples up to eight frames per training clip, averages spatial/temporal logits for classification, and trains a masked-feature predictor jointly. Validation clip accuracy selects `source.pt`.

## Run online adaptation

```bash
streamadapt run --data data/decoded --source outputs/source/source.pt --stream main --output outputs/full --method full --seed 13
```

The compact source backbone has three strided convolution/BatchNorm blocks, a dense class head and a convolutional masked-feature predictor. Its causal frame-window prediction averages dense logits over frames available so far. This explicit frame-CNN convention supplies dense features and predictions for the paper's state-transition method without requiring an external video checkpoint.

Only BatchNorm affine parameters adapt. Running statistics, classifier, feature predictor and convolution kernels remain fixed in StreamAdapt. A candidate makes three gradient steps using entropy and the motion/feature/track objective on the last sixteen frames. It then freezes. No further gradient update occurs while its decision is pending.

Evidence combines three losses: Jensen–Shannon disagreement after backward optical-flow alignment, feature reconstruction on a fixed 25% spatial mask, and confidence decrease on photometrically stable tracks. Farneback flow maps the current frame into the previous frame. Invalid warp positions are excluded. Mask randomness is identical across states for a given frame index. The first evidence comparison uses two strictly post-proposal frames, so neither member of the frame pair belongs to that candidate's adaptation window.

The advantage is committed loss minus candidate loss. Fixed Gaussian evidence densities give the log likelihood ratio. Acceptance and rejection thresholds are `log((1-beta)/alpha)` and `log(beta/(1-alpha))`. Their parameters are declared controls, not automatically estimated from evaluation labels. Dependence between video frames means these nominal tolerances do not themselves establish an achieved false-commit probability.

After a commit, the previous deployed state becomes the recovery anchor. A one-sided cumulative sum of live minus anchor proxy loss, with a drift allowance, triggers restoration. A new commit never overwrites the anchor with the newly deployed weights.

| Option | Default | Meaning |
|---|---:|---|
| `--window`, `--proposal-interval` | 16, 16 | Causal adaptation context and minimum proposal spacing |
| `--adapt-steps`, `--adapt-lr` | 3, 0.0001 | Candidate affine-parameter optimization |
| `--null-mean`, `--alternative-mean` | 0, 0.01 | Fixed Gaussian advantage means |
| `--evidence-std` | 0.1 | Common evidence standard deviation |
| `--alpha`, `--beta` | 0.05, 0.1 | Sequential boundary controls |
| `--motion-weight`, `--feature-weight`, `--track-weight` | 1, 0.1, 0.1 | Prequential evidence components |
| `--rollback-threshold`, `--rollback-drift` | 1, 0.001 | Post-commit cumulative degradation control |
| `--audit-frames` | 30 | Strictly future frame-pair observations for commit harm reporting |
| `--recovery-window`, `--recovery-tolerance` | 50, 0.05 | Accuracy window and recovery-band margin |

Labels are used only by source training, source validation and the online reporting layer. Candidate optimization, evidence, commitment and rollback do not read action labels.

## Comparisons and ablations

```bash
streamadapt run --data data/decoded --source outputs/source/source.pt --output outputs/frozen --method source
streamadapt run --data data/decoded --source outputs/source/source.pt --output outputs/tent --method tent
streamadapt run --data data/decoded --source outputs/source/source.pt --output outputs/immediate --method immediate
streamadapt run --data data/decoded --source outputs/source/source.pt --output outputs/entropy --method entropy_only
streamadapt run --data data/decoded --source outputs/source/source.pt --output outputs/no_motion --method no_motion
streamadapt run --data data/decoded --source outputs/source/source.pt --output outputs/fixed --method fixed_window
streamadapt run --data data/decoded --source outputs/source/source.pt --output outputs/no_rollback --method no_rollback
```

`source` never changes. `tent` uses entropy minimization on BatchNorm affine parameters with current-window batch statistics and immediate deployment. `immediate` retains the full candidate objective but removes future screening. `entropy_only` changes evidence alone, `no_motion` removes flow alignment from evidence, and `fixed_window` replaces sequential stopping with the sign majority after `--decision-window` observations. `no_rollback` retains future screening and removes anchor restoration. Keep source weights, stream order, resizing, stride and proposal schedule fixed across comparisons.

## Outputs and metrics

`metrics.json` reports live frame-level accuracy, cumulative labeled cross-entropy regret against the fixed source comparator, harmful update rate, mean commitment delay, recovery frames and complete replay FPS. Regret is a cumulative sum rather than a normalized benchmark-pool value. Recovery compares a post-boundary rolling window against the preceding window's accuracy minus the registered tolerance; source recovery is omitted because it performs no update.

Harmful updates are evaluated by frozen before/after states on the next `audit_frames` future frame pairs using the same prequential proxy. These audits do not control the learner. Commits too near the end remain unresolved and are excluded from the rate with their count reported. A stream without completed commits has a null harm rate. FPS includes decoding the prepared arrays, live/source prediction, adaptation, optical flow, evidence and commit audits, so it reflects the full reporting pipeline rather than an isolated backbone pass.

`stream.json` retains every issued prediction, pending status and gate score. `events.json` records proposal/commit timing, rejections and rollback. `commit_audits.json` contains completed paired harm assessments. `deployed.pt` stores the final deployed parameters; `source.pt` remains unchanged.

```bash
streamadapt plot --runs outputs/full outputs/frozen outputs/tent outputs/immediate --output outputs/figures
```

Run separate corruption orderings and streams with the same predeclared settings to compare transient and persistent shifts. Generated figures use only metrics produced by those runs.
