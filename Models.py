import torch
from torch import nn
from torch.nn import GRUCell, RNNCell
from beam_serach import BeamSearch

"""
All models must meet a few requirements
    1. They must have an init_model method that takes
    input_size and output_size as arguments
    2. They must have a forward method that takes inputs and hidden
    as arguments and returns output and hidden for one time step
    3. They must have a cell attribute that is the recurrent cell
    4. They must have a readout attribute that is the output layer
    (mapping from latent to output)
"""


class GRU_RNN(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None, latent_ic_var=0.05, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var
        self.recurrent_dropout = RecurrentDropout(dropout)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        hidden = self.recurrent_dropout(hidden)
        output = self.readout(hidden)
        return output, hidden
    
class GRU_RNN_StructPred(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None, latent_ic_var=0.05):
        super().__init__()
        # ANN Initialization
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(torch.zeros(latent_size), requires_grad=True)
        self.latent_ic_var = latent_ic_var


    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden
    
    def struct_forward(self, inputs, hidden):
        # ANN forward pass
        output, hidden = self.forward(inputs, hidden)
        # Beam search forward pass

        return output, hidden
    

class RecurrentDropout(nn.Module):
    def __init__(self, p):
        super(RecurrentDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = x.new(x.size()).bernoulli_(1 - self.p).div_(1 - self.p)
        return x * mask
