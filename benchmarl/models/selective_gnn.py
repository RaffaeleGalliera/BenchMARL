from dataclasses import MISSING, dataclass
from typing import Optional, Type
import inspect
import warnings
import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn, Tensor
from math import prod

from . import ModelConfig
from .gnn import Gnn, _RelVel, _get_edge_index
from tensordict import TensorDictBase
from tensordict.utils import _unravel_key_to_tuple


def _batch_from_dense_to_ptg_with_actions(
    x: Tensor,
    edge_index: Optional[Tensor],
    self_loops: bool,
    pos: Optional[Tensor] = None,
    vel: Optional[Tensor] = None,
    actions: Optional[Tensor] = None,
    num_actions: int = 4,
    edge_radius: Optional[float] = None,
    add_group_actions_to_node_features: bool = False
) -> torch_geometric.data.Batch:
    batch_size = prod(x.shape[:-2])
    n_agents = x.shape[-2]
    x = x.view(-1, x.shape[-1])
    if pos is not None:
        pos = pos.view(-1, pos.shape[-1])
    if vel is not None:
        vel = vel.view(-1, vel.shape[-1])

    b = torch.arange(batch_size, device=x.device)

    # Create initial graphs
    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * n_agents, n_agents, device=x.device)
    graphs.batch = torch.repeat_interleave(b, n_agents)
    graphs.x = x
    graphs.pos = pos
    graphs.vel = vel

    # add chosen action to node features
    if actions is not None and add_group_actions_to_node_features:
        # Shape of 'actions' is [batch_size * n_agents].
        # Convert it to one-hot encoding and concatenate.
        actions_oh = F.one_hot(actions, num_classes=num_actions).float()
        graphs.x = torch.cat([graphs.x, actions_oh], dim=-1)

    # Handle edge construction
    if edge_index is not None:
        n_edges = edge_index.shape[1]
        # Tensor of shape [batch_size * n_edges]
        # in which edges corresponding to the same graph have the same index.
        batch = torch.repeat_interleave(b, n_edges)
        # Edge index for the batched graphs of shape [2, n_edges * batch_size]
        # we sum to each batch an offset of batch_num * n_agents to make sure that
        # the adjacency matrices remain independent
        batch_edge_index = edge_index.repeat(1, batch_size) + batch * n_agents
        graphs.edge_index = batch_edge_index
    else:
        if pos is None:
            raise RuntimeError("from_pos topology needs positions as input")
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            pos, batch=graphs.batch, r=edge_radius, loop=self_loops
        )

    # Add action-based edges
    if actions is not None:
        actions = actions.view(-1)  # Flatten actions
        group = torch.repeat_interleave(b, n_agents) * num_actions + actions  # Global group ID

        # Get unique groups and inverse indices
        unique_groups, inverse_indices = group.unique(return_inverse=True)

        # Sort nodes by group
        sorted_indices = torch.argsort(inverse_indices)
        sorted_inverse_indices = inverse_indices[sorted_indices]

        # Compute group sizes and offsets
        group_sizes = torch.bincount(sorted_inverse_indices)
        group_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=x.device), group_sizes[:-1]]), dim=0)

        # Precompute total number of edges
        if self_loops:
            num_edges_per_group = group_sizes * group_sizes
        else:
            num_edges_per_group = group_sizes * (group_sizes - 1)
        total_num_edges = num_edges_per_group.sum().item()

        # Preallocate edge index tensors
        idx_i = torch.empty(total_num_edges, dtype=torch.long, device=x.device)
        idx_j = torch.empty(total_num_edges, dtype=torch.long, device=x.device)

        ptr = 0
        for i in range(len(group_sizes)):
            start = group_offsets[i].item()
            end = start + group_sizes[i].item()
            nodes = sorted_indices[start:end]
            n = nodes.size(0)
            if n > 1 or (n == 1 and self_loops):
                idx_i_g = nodes.repeat_interleave(n)
                idx_j_g = nodes.repeat(n)
                if not self_loops:
                    mask = idx_i_g != idx_j_g
                    idx_i_g = idx_i_g[mask]
                    idx_j_g = idx_j_g[mask]
                num_edges = idx_i_g.size(0)
                idx_i[ptr:ptr + num_edges] = idx_i_g
                idx_j[ptr:ptr + num_edges] = idx_j_g
                ptr += num_edges

        # Truncate idx_i and idx_j in case some groups have n <=1
        idx_i = idx_i[:ptr]
        idx_j = idx_j[:ptr]

        if idx_i.numel() > 0:
            edge_index_action = torch.stack([idx_i, idx_j], dim=0)
            graphs.edge_index = torch.cat([graphs.edge_index, edge_index_action], dim=1)

    # Apply graph transforms for edge attributes
    if pos is not None:
        graphs.edge_attr = None  # Reset edge_attr before recomputing
        graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
        graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
    if vel is not None:
        graphs = _RelVel()(graphs)

    return graphs


class SelectiveGnn(Gnn):
    def __init__(
        self,
        topology: str,
        self_loops: bool,
        gnn_class: Type[torch_geometric.nn.MessagePassing],
        gnn_kwargs: Optional[dict],
        position_key: Optional[str],
        exclude_pos_from_node_features: Optional[bool],
        velocity_key: Optional[str],
        edge_radius: Optional[float],
        pos_features: Optional[int],
        vel_features: Optional[int],
        num_actions: int = 4,
        add_group_actions_to_node_features: bool = False,
        **kwargs,
    ):
        self.topology = topology
        self.self_loops = self_loops
        self.position_key = position_key
        self.velocity_key = velocity_key
        self.exclude_pos_from_node_features = exclude_pos_from_node_features
        self.edge_radius = edge_radius
        self.pos_features = pos_features
        self.vel_features = vel_features
        self.add_group_actions_to_node_features = add_group_actions_to_node_features

        super(Gnn, self).__init__(**kwargs)

        if self.pos_features > 0:
            self.pos_features += 1  # We will add also 1-dimensional distance
        self.edge_features = self.pos_features + self.vel_features
        self.input_features = sum(
            [
                spec.shape[-1]
                for key, spec in self.input_spec.items(True, True)
                if _unravel_key_to_tuple(key)[-1] not in (position_key, velocity_key)
            ]
        )  # Input keys
        if self.position_key is not None and not self.exclude_pos_from_node_features:
            self.input_features += self.pos_features - 1
        if self.velocity_key is not None:
            self.input_features += self.vel_features

        # Add group actions to input features
        self.action_head_input_features = self.input_features
        if self.add_group_actions_to_node_features:
            self.input_features += num_actions

        self.output_features = self.output_leaf_spec.shape[-1]

        if gnn_kwargs is None:
            gnn_kwargs = {}
        gnn_kwargs.update(
            {"in_channels": self.input_features, "out_channels": self.output_features}
        )
        self.gnn_supports_edge_attrs = (
            "edge_dim" in inspect.getfullargspec(gnn_class).args
        )
        if (
            self.position_key is not None or self.velocity_key is not None
        ) and not self.gnn_supports_edge_attrs:
            warnings.warn(
                "Position key or velocity key provided but GNN class does not support edge attributes. "
                "These keys will not be used for computing edge features."
            )
        if (
            position_key is not None or velocity_key is not None
        ) and self.gnn_supports_edge_attrs:
            gnn_kwargs.update({"edge_dim": self.edge_features})

        self.gnns = nn.ModuleList(
            [
                gnn_class(**gnn_kwargs).to(self.device)
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )
        self.edge_index = _get_edge_index(
            topology=self.topology,
            self_loops=self.self_loops,
            device=self.device,
            n_agents=self.n_agents,
        )
        self._full_position_key = None
        self._full_velocity_key = None

        # Add group selection action head
        self.num_actions = num_actions
        self.action_head = nn.Linear(self.action_head_input_features, self.num_actions).to(self.device)


    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        input_features = [
            tensordict.get(in_key)
            for in_key in self.in_keys
            if _unravel_key_to_tuple(in_key)[-1]
            not in (self.position_key, self.velocity_key)
        ]

        # Retrieve position
        if self.position_key is not None:
            if self._full_position_key is None:
                self._full_position_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.position_key
                )
            pos = tensordict.get(self._full_position_key)
            if not self.exclude_pos_from_node_features:
                input_features.append(pos)
        else:
            pos = None

        # Retrieve velocity
        if self.velocity_key is not None:
            if self._full_velocity_key is None:
                self._full_velocity_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.velocity_key
                )
            vel = tensordict.get(self._full_velocity_key)
            input_features.append(vel)
        else:
            vel = None

        # Concatenate input features
        input = torch.cat(input_features, dim=-1)
        batch_size = input.shape[:-2]

        # Flatten input for action head
        x = input.view(-1, input.shape[-1]).to(self.device)

        # Compute actions
        action_logits = self.action_head(x)
        action_probs = F.gumbel_softmax(action_logits, tau=1, hard=True)
        actions = action_probs.argmax(dim=-1)  # Shape: [batch_size * n_agents]

        # Build the graph with both radius-based and action-based edges
        graph = _batch_from_dense_to_ptg_with_actions(
            x=input.to(self.device),
            edge_index=self.edge_index.to(self.device) if self.edge_index is not None else None,
            pos=pos.to(self.device) if pos is not None else None,
            vel=vel.to(self.device) if vel is not None else None,
            actions=actions,
            num_actions=self.num_actions,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
            add_group_actions_to_node_features=self.add_group_actions_to_node_features
        )

        # Proceed with GNN forward pass
        forward_gnn_params = {
            "x": graph.x,
            "edge_index": graph.edge_index,
        }
        if (
            (self.position_key is not None or self.velocity_key is not None)
            and self.gnn_supports_edge_attrs
            and graph.edge_attr is not None
        ):
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
                        for i, gnn in enumerate(self.gnns)
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
                        for i, gnn in enumerate(self.gnns)
                    ],
                    dim=-2,
                )
        else:
            res = self.gnns[0](**forward_gnn_params).view(
                *batch_size, self.n_agents, self.output_features
            )
            if self.centralised:
                res = res.mean(dim=-2)  # Mean pooling

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class SelectiveGnnConfig(ModelConfig):
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

    num_actions: int = 4  # Number of discrete actions for each node
    add_group_actions_to_node_features: bool = False # Add group actions to node features

    @staticmethod
    def associated_class():
        return SelectiveGnn
