"""Default locations, each overridable with an environment variable (and a CLI flag in the scripts).

  WASP_SR_DATA     SR benchmarks and DIV2K               default <repo>/datasets/sr
  WASP_IQA_DATA    TID2013 / KADID-10K / PIPAL           default <repo>/datasets/iqa
  WASP_CKPT        released fine-tuned checkpoints       default <repo>/checkpoints
  WASP_PRETRAINED  official pretrained weights (cache)   default <repo>/pretrained
  WASP_RESULTS     caches, result JSON, tables, figures  default <repo>/results
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(name, default)))


SR_DATA = _env('WASP_SR_DATA', os.path.join(REPO, 'datasets', 'sr'))
IQA_DATA = _env('WASP_IQA_DATA', os.path.join(REPO, 'datasets', 'iqa'))
CKPT_DIR = _env('WASP_CKPT', os.path.join(REPO, 'checkpoints'))
PRETRAINED_DIR = _env('WASP_PRETRAINED', os.path.join(REPO, 'pretrained'))
RESULTS_DIR = _env('WASP_RESULTS', os.path.join(REPO, 'results'))

DIV2K_TRAIN_HR = os.path.join(SR_DATA, 'DIV2K', 'DIV2K_train_HR')
DIV2K_VALID_HR = os.path.join(SR_DATA, 'DIV2K', 'DIV2K_valid_HR')
