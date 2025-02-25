#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .cnn import Cnn, CnnConfig
from .common import (
    EnsembleModelConfig,
    Model,
    ModelConfig,
    SequenceModel,
    SequenceModelConfig,
)
from .deepsets import Deepsets, DeepsetsConfig
from .gnn import Gnn, GnnConfig
from .gru import Gru, GruConfig
from .lstm import Lstm, LstmConfig
from .mlp import Mlp, MlpConfig
from .selective_gnn import SelectiveGnnConfig
from .selective_gnn_two_layers import SelectiveGnnTwoLayersConfig
from .gnn_two_layers import GnnTwoLayersConfig
from .gumbel_selective_gnn import GumbelSelectiveGnnConfig, GumbelSelectiveGnn
from .gumbel_selective_gnn_two_layers import GumbelSelectiveGnnTwoLayersConfig, GumbelSelectiveGnnTwoLayers

classes = [
    "Mlp",
    "MlpConfig",
    "Gnn",
    "GnnConfig",
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
    "Gru",
    "GruConfig",
    "Lstm",
    "LstmConfig",
    "SelectiveGnn",
    "SelectiveGnnConfig",
    "SelectiveGnnTwoLayers",
    "SelectiveGnnTwoLayersConfig",
    "GnnTwoLayers",
    "GnnTwoLayersConfig",
    "GumbelSelectiveGnn",
    "GumbelSelectiveGnnConfig",
    "GumbelSelectiveGnnTwoLayers",
    "GumbelSelectiveGnnTwoLayersConfig",
]

model_config_registry = {
    "mlp": MlpConfig,
    "gnn": GnnConfig,
    "cnn": CnnConfig,
    "deepsets": DeepsetsConfig,
    "gru": GruConfig,
    "lstm": LstmConfig,
    "selective_gnn": SelectiveGnnConfig,
    "selective_gnn_two_layers": SelectiveGnnTwoLayersConfig,
    "gnn_two_layers": GnnTwoLayersConfig,
    "gumbel_selective_gnn": GumbelSelectiveGnnConfig,
    "gumbel_selective_gnn_two_layers": GumbelSelectiveGnnTwoLayersConfig
}
