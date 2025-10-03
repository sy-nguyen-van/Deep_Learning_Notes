## Repo overview

This repository is a small teaching / homework assignment for implementing and
training a two-layer neural network on CIFAR-10. Key entry points:

- `two_layer_net.py` — top-level notebook/script showing experiments: toy data
  checks, training loop examples, visualization and hyperparameter search.
- `lib/classifiers/neural_net.py` — the student-implemented `TwoLayerNet` class
  (forward, loss, backward, train, predict). This is the central code AI
  edits will touch most often.
- `lib/data_utils.py` — helpers to load CIFAR-10 and TinyImageNet; returns
  preprocessed NumPy arrays or dictionaries used by training code.
- `lib/gradient_check.py` — numerical gradient utilities used by tests.
- `lib/vis_utils.py` — small visualization helpers (grid visualization).

Files follow simple, educational patterns (NumPy-only code, no deep learning
framework). Parameters are stored in `self.params` dict on `TwoLayerNet`.

## What an AI agent should know first (big picture)

- The network architecture is a standard 2-layer fully-connected net with ReLU
  and a softmax loss. `TwoLayerNet` stores parameters in `self.params` with
  keys `W1`, `b1`, `W2`, `b2`. Shapes:
  - `W1`: (D, H), `b1`: (H,), `W2`: (H, C), `b2`: (C,)
- `two_layer_net.py` demonstrates the expected inputs / outputs for methods:
  - `loss(X, y=None, reg=0.0)` returns scores when `y is None` and
    `(loss, grads)` when `y` is provided.
  - `train(...)` returns a dict with `loss_history`, `train_acc_history`,
    `val_acc_history` (used by plotting code in the notebook).
- Data shapes: most helper code expects image data to be row-major flattened
  arrays for training (e.g. `X_train.reshape(num_training, -1)` in the
  notebook) while `lib/data_utils.get_CIFAR10_data` can also return channel-first
  arrays inside its dictionary form. Inspect how the caller arranges shapes
  before editing data preprocessing.

## Developer workflows / commands

- The project is primarily an interactive / notebook workflow. Start by
  running `two_layer_net.py` or opening `two_layer_net.ipynb` in Jupyter.
  Typical Python environment requirements are in `requirements.txt`.
- No build system or tests are provided. Useful dev steps:

```powershell
# Create virtualenv (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Run the demo script (runs non-notebook experiments)
python .\two_layer_net.py
```

Notes:
- The notebook contains example training runs with CIFAR-10. Download the
  dataset into `lib/datasets/cifar-10-batches-py` (there is a `get_datasets.sh`
  script in `lib/datasets` that can help on Unix-like systems).

## Project-specific conventions and patterns

- Minimal external deps: code is NumPy-based and educational; avoid adding
  heavy framework changes unless the assignment requires it.
- Parameter storage: always access weights through `self.params['W1']` etc.
  Gradients use the same key names in returned `grads` dict.
- Loss API: `loss(X, y=None, reg=0.0)` — calling with `y=None` must return the
  raw class scores matrix (N, C). When `y` is provided it must return
  `(loss, grads)` where `grads` contains numeric arrays for each parameter.
- Randomness: many examples set `np.random.seed(...)` in the notebook; preserve
  or control seeds when writing deterministic tests or small experiments.

## Common edit patterns and examples

- Implementing forward/backward: follow the pattern in `TwoLayerNet.loss` —
  compute forward (scores), if `y` is None return scores; otherwise compute
  softmax loss + L2 reg, then populate `grads` entries for `'W1','b1','W2','b2'`.
  Example usage: numeric gradient checks are performed with
  `lib.gradient_check.eval_numerical_gradient` in `two_layer_net.py`.
- Training loop: `train` uses mini-batch SGD. `iterations_per_epoch` is
  calculated as `max(num_train / batch_size, 1)` — note it uses Python 2-style
  division in some files; prefer using integer division or casting to int when
  editing for Python 3 clarity.

## Integration points & external dependencies

- Data: `lib/data_utils.load_CIFAR10` expects CIFAR-10 extracted under
  `lib/datasets/cifar-10-batches-py`. `two_layer_net.py` calls
  `lib.data_utils.load_CIFAR10` directly.
- imageio (in `requirements.txt`) is used by `lib/data_utils` for reading
  image files (tiny imagenet utilities).

## Things to watch for (edge cases found in code)

- Some code mixes shapes conventions (channel-first vs channel-last). The
  notebook reshapes images to row vectors for the simple `TwoLayerNet` code.
  Confirm caller expectations before changing data layout.
- A few places use Python 2-era idioms (e.g., `six.moves.cPickle` and
  version-detection in `data_utils`) — safe, but avoid introducing Python-3
  incompatible changes without adjusting imports.
- `train` uses `iterations_per_epoch = max(num_train / batch_size, 1)` which
  yields a float in Python 3; when used as a modulo divider (`it % iterations_per_epoch`)
  callers may expect an integer. Prefer casting to `int()` when modifying.

## Minimal checklist for changes an agent should run locally

- Install dependencies from `requirements.txt`.
- Run `python .\two_layer_net.py` and verify the toy checks and gradient
  checks in the script run without exceptions.
- If editing `TwoLayerNet.loss`, run the numeric gradient checks shown in
  `two_layer_net.py` (they call `eval_numerical_gradient`) and ensure relative
  errors are small (<1e-7 typical for this assignment).

## Where to look for examples

- Correct numeric gradient usage: `two_layer_net.py` lines that call
  `eval_numerical_gradient`.
- Visualization helper: `lib/vis_utils.visualize_grid` is used to display
  first-layer weights in the notebook.

---
If any part of these instructions is unclear or you'd like me to expand a
specific section (for example, adding a small automated test harness or
documenting more code examples), tell me which part and I'll iterate.
