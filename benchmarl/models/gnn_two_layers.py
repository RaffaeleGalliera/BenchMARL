from dataclasses import MISSING, dataclass
from typing import Optional, Type

import torch
import torch_geometric
from torch import nn
from torch_geometric.nn import Sequential

from . import ModelConfig
from .gnn import Gnn, _batch_from_dense_to_ptg
from tensordict import TensorDictBase
from tensordict.utils import _unravel_key_to_tuple


class GnnTwoLayers(Gnn):
    def __init__(self, num_actions: int = 4, **kwargs):
        super().__init__(num_actions, **kwargs)
        gnn_class = kwargs.get("gnn_class", MISSING)
        if gnn_class is MISSING:
            raise ValueError("gnn_class must be provided")
        gnn_kwargs = kwargs.get("gnn_kwargs", {})
        gnn_kwargs["in_channels"] = self.output_features
        self.models = nn.ModuleList(
            [
                Sequential(
                    'x, edge_index, edge_attr',
                    [
                        (self.gnns[0], 'x, edge_index, edge_attr -> x'),
                        nn.ReLU(inplace=True),
                        (gnn_class(**gnn_kwargs),'x, edge_index, edge_attr -> x'),
                    ]
                ).to(self.device)
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = [
            tensordict.get(in_key)
            for in_key in self.in_keys
            if _unravel_key_to_tuple(in_key)[-1]
               not in (self.position_key, self.velocity_key)
        ]

        # Retrieve position
        if self.position_key is not None:
            if self._full_position_key is None:  # Run once to find full key
                self._full_position_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.position_key
                )
                pos = tensordict.get(self._full_position_key)
                if pos.shape[-1] != self.pos_features - 1:
                    raise ValueError(
                        f"Position key in tensordict is {pos.shape[-1]}-dimensional, "
                        f"while model was configured with pos_features={self.pos_features - 1}"
                    )
            else:
                pos = tensordict.get(self._full_position_key)
            if not self.exclude_pos_from_node_features:
                input.append(pos)
        else:
            pos = None

        # Retrieve velocity
        if self.velocity_key is not None:
            if self._full_velocity_key is None:  # Run once to find full key
                self._full_velocity_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.velocity_key
                )
                vel = tensordict.get(self._full_velocity_key)
                if vel.shape[-1] != self.vel_features:
                    raise ValueError(
                        f"Velocity key in tensordict is {vel.shape[-1]}-dimensional, "
                        f"while model was configured with vel_features={self.vel_features}"
                    )
            else:
                vel = tensordict.get(self._full_velocity_key)
            input.append(vel)
        else:
            vel = None

        input = torch.cat(input, dim=-1)
        batch_size = input.shape[:-2]

        graph = _batch_from_dense_to_ptg(
            x=input,
            edge_index=self.edge_index,
            pos=pos,
            vel=vel,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
        )
        forward_gnn_params = {
            "x": graph.x,
            "edge_index": graph.edge_index,
        }
        if (
                self.position_key is not None or self.velocity_key is not None
        ) and self.gnn_supports_edge_attrs:
            forward_gnn_params.update({"edge_attr": graph.edge_attr})

        if not self.share_params:
            if not self.centralised:
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params).view(
                            *batch_size,
                            self.n_agents,
                            self.output_features,
                        )[..., i, :]
                        for i, gnn in enumerate(self.models)
                    ],
                    dim=-2,
                )
            else:
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params)
                        .view(
                            *batch_size,
                            self.n_agents,
                            self.output_features,
                        )
                        .mean(dim=-2)  # Mean pooling
                        for i, gnn in enumerate(self.models)
                    ],
                    dim=-2,
                )

        else:
            res = self.models[0](**forward_gnn_params).view(
                *batch_size, self.n_agents, self.output_features
            )
            if self.centralised:
                res = res.mean(dim=-2)  # Mean pooling

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class GnnTwoLayersConfig(ModelConfig):
    """Dataclass config for a SelectiveGNN."""

    topology: str = MISSING
    self_loops: bool = MISSING

    gnn_class: Type[torch_geometric.nn.MessagePassing] = MISSING
    gnn_kwargs: Optional[dict] = None

    position_key: Optional[str] = None
    pos_features: Optional[int] = 0
    velocity_key: Optional[str] = None
    vel_features: Optional[int] = 0
    exclude_pos_from_node_features: Optional[bool] = None
    edge_radius: Optional[float] = None

    @staticmethod
    def associated_class():
        return GnnTwoLayers