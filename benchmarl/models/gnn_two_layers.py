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
    def __init__(
            self,
             _get_pos_from_features: bool = False,  ### NEW
             **kwargs
    ):
        """
        :param num_actions: Number of discrete actions (passed to super)
        :param _get_pos_from_features: If True, skip retrieving position from tensordict
                                       and extract (x, y) from the last two channels
                                       of the input feature array.
        :param kwargs: Additional arguments (like gnn_class, gnn_kwargs, etc.)
        """
        super().__init__(**kwargs)
        self._get_pos_from_features = _get_pos_from_features  ### NEW

        gnn_class = kwargs.get("gnn_class", MISSING)
        if gnn_class is MISSING:
            raise ValueError("gnn_class must be provided")
        gnn_kwargs = kwargs.get("gnn_kwargs", {})

        # The first GNN layer is already in self.gnns[0], from the parent Gnn constructor.
        # We'll build a second GNN layer in a sequential pipeline:
        gnn_kwargs["in_channels"] = self.output_features

        # If we rely on edge_attr (pos/vel), we must pass 'edge_attr' as well:
        input_args = "x, edge_index, edge_attr" if self.topology == "from_pos" else "x, edge_index"
        self.group_key = kwargs.get("agent_group")
        self.models = nn.ModuleList(
            [
                Sequential(
                    input_args,
                    [
                        (self.gnns[0], f"{input_args} -> x"),
                        nn.ReLU(inplace=True),
                        (gnn_class(**gnn_kwargs), f"{input_args} -> x"),
                    ]
                ).to(self.device)
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Build a graph from node features, possibly parse pos from them if _get_pos_from_features=True."""
        # Collect the standard in_key features (excluding position_key and velocity_key):
        input_list = [
            tensordict.get(in_key)
            for in_key in self.in_keys
            if _unravel_key_to_tuple(in_key)[-1]
            not in (self.position_key, self.velocity_key)
        ]

        # Default pos, vel to None
        pos = None
        vel = None

        ########################################################################
        # 1) If we are NOT extracting position from the last channels,
        #    proceed to retrieve position/velocity from the tensordict.
        ########################################################################
        if not self._get_pos_from_features:
            # Retrieve pos from tensordict if position_key is set
            if self.position_key is not None:
                if self._full_position_key is None:
                    self._full_position_key = self._get_key_terminating_with(
                        list(tensordict.keys(True, True)), self.position_key
                    )
                    pos_ = tensordict.get(self._full_position_key)
                    if pos_.shape[-1] != self.pos_features - 1:
                        raise ValueError(
                            f"Position key in tensordict is {pos_.shape[-1]}-dim, "
                            f"while model was configured with pos_features={self.pos_features - 1}"
                        )
                else:
                    pos_ = tensordict.get(self._full_position_key)

                if not self.exclude_pos_from_node_features:
                    input_list.append(pos_)
                pos = pos_

            # Retrieve vel from tensordict if velocity_key is set
            if self.velocity_key is not None:
                if self._full_velocity_key is None:
                    self._full_velocity_key = self._get_key_terminating_with(
                        list(tensordict.keys(True, True)), self.velocity_key
                    )
                    vel_ = tensordict.get(self._full_velocity_key)
                    if vel_.shape[-1] != self.vel_features:
                        raise ValueError(
                            f"Velocity key in tensordict is {vel_.shape[-1]}-dim, "
                            f"while model was configured with vel_features={self.vel_features}"
                        )
                else:
                    vel_ = tensordict.get(self._full_velocity_key)

                input_list.append(vel_)
                vel = vel_

        ########################################################################
        # 2) Otherwise, if we DO want to parse position from the last features,
        #    we skip reading from tensordict, and we will slice them from input.
        ########################################################################
        # (We'll do that after we concatenate below.)
        ########################################################################

        # Merge all input features into one tensor
        # shape = [*batch_size, n_agents, combined_feature_dim]
        input_tensor = torch.cat(input_list, dim=-1)
        batch_size = input_tensor.shape[:-2]

        # If we are *extracting pos from features*:
        if self._get_pos_from_features:
            original_features = tensordict.get(self.group_key, "observation")['observation']
            # e.g., we want the last 2 dims to be x,y position
            if original_features.shape[-1] < 2:
                raise ValueError(
                    "Original Features does not have enough feature dims to slice out (x,y) position."
                )
            pos = original_features[..., 0, 0, -2:]


        # Build the graph (we rely on 'pos' and 'vel' for edge_attr if topology=="from_pos")
        graph = _batch_from_dense_to_ptg(
            x=input_tensor,
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

        ########################################################################
        # Run the 2-layer GNN pipeline (self.models) over the built graph
        ########################################################################

        if not self.share_params:
            if not self.centralised:
                # Each agent uses its own GNN
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params).view(*batch_size, self.n_agents, self.output_features)[..., i, :]
                        for i, gnn in enumerate(self.models)
                    ],
                    dim=-2,
                )
            else:
                # Each agent uses its own GNN, then we average across the agent dimension
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
    """Dataclass config for a 2-layer GNN."""

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

    _get_pos_from_features: bool = False  ### NEW

    @staticmethod
    def associated_class():
        return GnnTwoLayers
