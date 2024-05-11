from torch import nn


class Mnist_Model_MLP(nn.Module):
    def __init__(self, mlp_hidden: list = [784, 256, 128, 10]):
        super().__init__()
        assert mlp_hidden[0] == 784 and mlp_hidden[-1] == 10

        layers = []
        for item in zip(mlp_hidden[:-1], mlp_hidden[1:]):
            layers.extend([nn.Linear(*item), nn.ReLU()])

        # remove the last relu activation function
        layers = layers[:-1]
        self.mlp = nn.Sequential(*nn.ModuleList(layers))

    def forward(self, x):
        return self.mlp(x)
