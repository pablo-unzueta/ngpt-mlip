import torch
import torch.nn as nn
from cbp_linear import CBPLinear


torch.manual_seed(42)


class Block(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_hidden: int,
        num_targets: int,
        act_type: str = "relu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.act = nn.ReLU()


class CBPSimpleMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_hidden: int,
        num_targets: int,
        replacement_rate: float = 0.0,
        maturity_threshold: int = 100,
        init: str = "default",
        decay_rate: float = 0.0,
        act_type: str = "relu",
        util_type: str = "contribution",
        ln_layer: bool = False,
        bn_layer: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_targets)
        self.act = nn.ReLU()

        # Initialize CBP layers

        self.cbp1 = CBPLinear(
            in_layer=self.fc1,
            out_layer=self.fc2,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            init=init,
        )
        self.cbp2 = CBPLinear(
            in_layer=self.fc2,
            out_layer=self.fc3,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            init=init,
        )

        self.layers = nn.ModuleList()
        self.layers.append(self.fc1)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc2)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc3)

        self.act_type = act_type

    def predict(self, x):
        # Input passes through CBP layers after the non-linearities
        x1 = self.cbp1(self.act(self.fc1(x)))
        x2 = self.cbp2(self.act(self.fc2(x1)))
        x3 = self.fc3(x2)
        return x3, [x1, x2]

    def forward(self, x):
        output, _ = self.predict(x)
        return output
