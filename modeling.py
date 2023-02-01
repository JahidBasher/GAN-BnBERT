from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        noise_size=100,
        output_size=768,
        hidden_sizes=[768],
        dropout_rate=0.1
    ):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


class Discriminator(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_sizes=[768],
        num_labels=2,
        dropout_rate=0.1
    ):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])

        self.layers = nn.Sequential(*layers) 
        self.logit = nn.Linear(hidden_sizes[-1], num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs
