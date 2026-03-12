import torch.nn as nn


class TitanicModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        layers = []

        input_dim = config["input_dim"]

        for hidden in config["hidden_layers"]:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["dropout"]))
            input_dim = hidden

        layers.append(nn.Linear(input_dim, config["output_dim"]))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)