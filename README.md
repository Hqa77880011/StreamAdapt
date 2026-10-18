# StreamAdapt

Implementation of **StreamAdapt: Evidence-Gated Test-Time Adaptation for Reliable Model-State Transitions in Nonstationary Video Streams**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

The included finite generator renders train/validation/stream videos with disjoint source groups. For external clips use the schema below and your licensed collection.

```bash
python finite_data.py --output data/generated --seed 13
```

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
streamadapt prepare --manifest data/generated/videos.csv --output data/decoded --size 128 --stride 1
```

## Training and inference

```bash
streamadapt train --data data/decoded --output outputs/source --epochs 50 --frames 8 --seed 13
streamadapt run --data data/decoded --source outputs/source/source.pt --stream main --output outputs/full --method full --seed 13
```

## Evaluation

The online run records predictions before adaptation, gate decisions, commits and rollbacks. Labels are used only for reporting.

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

## Outputs

`outputs/source/source.pt` stores the source checkpoint. Online outputs retain `metrics.json`, `events.json`, `stream.json`, `commit_audits.json` and `deployed.pt`. The existing pipeline combines generation, source fitting and full/source/immediate comparisons:

```bash
python reproduce.py --plan
python reproduce.py --output outputs/finite-setting
streamadapt plot --runs outputs/full outputs/frozen outputs/tent outputs/immediate --output outputs/figures
```
