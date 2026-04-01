================================================================================
  Petri net discovery + ASFormer (train / predict / eval)
================================================================================

This folder is meant to live beside (or anywhere) your ASFormer checkout.
ASFormer's main.py must be run from the ASFormer project root (where main.py,
eval.py, and data/ live).

-------------------------------------------------------------------------------
Requirements
-------------------------------------------------------------------------------

- Python 3.9+ (3.10 recommended for PyTorch stacks)
- Packages: pm4py, pandas, torch, and the rest of ASFormer's dependencies
- A clone of ASFormer (e.g. ASFormer-main) with dataset data prepared under:
    <ASFormer>/data/<dataset>/groundTruth/
    <ASFormer>/data/<dataset>/mapping.txt
    (plus features, splits, etc. as ASFormer expects)

Install dependencies in the same environment you use for training.

-------------------------------------------------------------------------------
Paths (replace with yours)
-------------------------------------------------------------------------------

  PETRI_ROOT     = directory containing this repo (petri_net_inductive_miner.py,
                   run_pipeline.py, petri_adapter.py, etc.)

  ASFORMER_ROOT  = directory containing ASFormer's main.py (e.g. ~/ASFormer-main)

  DATASET_ROOT   = ASFormer dataset folder, e.g.
                   ASFORMER_ROOT/data/50salads
                   ASFORMER_ROOT/data/breakfast

-------------------------------------------------------------------------------
One-command pipeline (recommended)
-------------------------------------------------------------------------------

From PETRI_ROOT:

  python run_pipeline.py \
    --asformer-root "/path/to/ASFormer-main" \
    --dataset 50salads \
    --dataset-root "/path/to/ASFormer-main/data/50salads" \
    --steps discover,train,predict,eval \
    --petri-net-file "/path/to/save/50salads_net.pnml"

Notes:
- --steps all   is the same as discover,train,predict,eval
- If --dataset-root is omitted, it defaults to:
    <asformer-root>/data/<dataset>
- If --petri-net-file is set and the discover step is included, run_pipeline
  saves the mined net to that path and then runs train/predict with
  --petri_mode load so training uses the same PNML (not a second discovery).

Breakfast example (silence label SIL is filtered in the miner):

  python run_pipeline.py \
    --asformer-root "/path/to/ASFormer-main" \
    --dataset breakfast \
    --steps discover,train,predict,eval \
    --petri-net-file "/path/to/save/breakfast_net.pnml"

Train only after you already have a PNML (skip discover):

  python run_pipeline.py \
    --asformer-root "/path/to/ASFormer-main" \
    --dataset 50salads \
    --steps train,predict,eval \
    --petri-mode load \
    --petri-net-file "/path/to/50salads_net.pnml"

Optional flags:
  --split 1
  --weight-disagreement 1.5
  --petri-root "/path/to/this/petri/repo"   (default: directory of run_pipeline.py)

-------------------------------------------------------------------------------
Discover the Petri net only (miner)
-------------------------------------------------------------------------------

From PETRI_ROOT:

  python petri_net_inductive_miner.py \
    --dataset 50salads \
    --dataset-root "/path/to/ASFormer-main/data/50salads" \
    --output-pnml "/path/to/50salads_net.pnml"

- --dataset breakfast  filters label "SIL";  --dataset 50salads  filters "background".
- --dataset-root must be a real path (must contain groundTruth/ and mapping.txt).
  Do not use placeholder paths like "/path/to/...".

-------------------------------------------------------------------------------
Manual ASFormer commands (without run_pipeline.py)
-------------------------------------------------------------------------------

From ASFORMER_ROOT, same Python environment:

Train (discover net inside main.py if --petri_mode discover):

  python main.py --dataset 50salads --split 1 \
    --dataset_root "/path/to/ASFormer-main/data/50salads" \
    --use_petri --petri_path "/path/to/petri/repo" \
    --weight_disagreement 1.5 --petri_mode discover

Train loading an existing PNML:

  python main.py --dataset 50salads --split 1 \
    --dataset_root "/path/to/ASFormer-main/data/50salads" \
    --use_petri --petri_path "/path/to/petri/repo" \
    --weight_disagreement 1.5 --petri_mode load \
    --petri_net_file "/path/to/50salads_net.pnml"

Predict (use the same flags as training):

  python main.py --action predict --dataset 50salads --split 1 \
    --dataset_root "/path/to/ASFormer-main/data/50salads" \
    --use_petri --petri_path "/path/to/petri/repo" \
    --weight_disagreement 1.5 --petri_mode load \
    --petri_net_file "/path/to/50salads_net.pnml"

Eval:

  python eval.py --dataset 50salads --split 1

-------------------------------------------------------------------------------
Shell wrapper (run_pipeline.sh)
-------------------------------------------------------------------------------

There is also run_pipeline.sh for conda-based workflows. Edit ASFORMER_ROOT,
DATASET, and PETRI_ROOT if needed. The shell script may be older than
run_pipeline.py; prefer run_pipeline.py for full parity with --dataset-root and
--petri-net-file.

-------------------------------------------------------------------------------
Troubleshooting
-------------------------------------------------------------------------------

- ModuleNotFoundError (pm4py, pandas, torch): install in the active env.
- Predictions look wrong vs training: train and predict must use the same
  --use_petri, --petri_mode, and --petri_net_file.
- NumPy 2: if eval.py fails on np.float, replace np.float with np.float64 in
  eval.py (or pin numpy<2).
- macOS SSL errors when downloading the optional demo XES: only happens if no
  dataset path is found; fix by pointing --dataset-root to a real folder.

================================================================================
