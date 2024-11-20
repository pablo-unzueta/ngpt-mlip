import torch_geometric
import torch
from torch import nn
from torch_geomtric.nn import GATConv
from torch import Tensor
from torch_geometric.nn import radius_graph


class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """

    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AttentionBlock(nn.Module):
    def __init__(self, num_features: int, num_targets: int, heads=8):
        super().__init__()
        self.conv = GATConv(
            num_features, num_targets, heads=heads
        )  # -> (num_targets * heads)
        self.ln = nn.LayerNorm(num_targets * heads)
        self.ff = nn.Linear(num_targets * heads, num_features)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.ln(out)
        out = self.ff(out)
        x = x + out
        return x


class Model(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_features: int = 11,
        num_targets: int = 19,
        heads: int = 8,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        num_gaussians: int = 50,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0)
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(num_features, num_targets, heads)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, data):
        h = self.embedding(data.z)
        edge_index, edge_weight = self.interaction_graph(data.pos, data.batch)
        edge_attr = self.distance_expansion(edge_weight)

        for block in self.blocks:
            h = h + block(h, edge_index, edge_weight, edge_attr)

        return x


# if __name__ == __main__():
