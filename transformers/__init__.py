# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "2.5.1"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_tf_available,
    is_torch_available,
)
# Model Cards
from .modelcard import ModelCard


# Pipelines
# from .pipelines import (
#     CsvPipelineDataFormat,
#     FeatureExtractionPipeline,
#     FillMaskPipeline,
#     JsonPipelineDataFormat,
#     NerPipeline,
#     PipedPipelineDataFormat,
#     Pipeline,
#     PipelineDataFormat,
#     QuestionAnsweringPipeline,
#     TextClassificationPipeline,
#     TokenClassificationPipeline,
#     pipeline,
# )

from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from .tokenization_utils import PreTrainedTokenizer
from .modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# Modeling
if is_torch_available():
    from .modeling_gpt2 import (
        GPT2PreTrainedModel,
        GPT2Model,
        GPT2LMHeadModel,
        GPT2DoubleHeadsModel,
        load_tf_weights_in_gpt2,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
    )

    from .modeling_gpt2_with_memory import GPT2MemoryLMHeadModel

    # Optimization
    from .optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )


    # Optimization
    # from .optimization_tf import WarmUp, create_optimizer, AdamWeightDecay, GradientAccumulator


if not is_tf_available() and not is_torch_available():
    logger.warning(
        "Neither PyTorch nor TensorFlow >= 2.0 have been found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
