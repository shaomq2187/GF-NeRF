import torch
from torch import nn
class MLPNetwork(nn.Module):
	"""
	Neural network.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a tensor of shape `[:, n_output_dims]`.

	The output tensor can be either of type `torch.float` or `torch.half`,
	depending on which performs better on the system.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	n_output_dims : `int`
		Determines the shape of output tensors as `[:, n_output_dims]`
	network_config: `dict`
		Configures the neural network. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	"""
	def __init__(self, n_input_dims, n_output_dims, network_config, seed=1337):
		super().__init__()
		self.n_input_dims = n_input_dims
		self.n_output_dims = n_output_dims
		self.network_config = network_config
		self.n_neurons = network_config['n_neurons']
		self.n_hidden_layers = network_config['n_hidden_layers']
		self.output_activation = nn.Sigmoid() if network_config['output_activation'] == 'Sigmoid' else None
		assert network_config['activation'] == "ReLU"

		self.layers = nn.ModuleList()
		self.relu = nn.ReLU()
		for i in range(self.n_hidden_layers + 1):
			if i == 0:
				self.layers.append(nn.Linear(self.n_input_dims,self.n_neurons))
			elif i == self.n_hidden_layers:
				self.layers.append(nn.Linear(self.n_neurons,self.n_output_dims))
			else:
				self.layers.append(nn.Linear(self.n_neurons,self.n_neurons))
		
	def forward(self,x):
		for i in range(self.n_hidden_layers + 1):
			layer = self.layers[i]

			if i == self.n_hidden_layers:
				# output layer
				x = layer(x)
				if self.output_activation is not None:
					x = self.output_activation(x)
			else:
				x = layer(x)
				x = self.relu(x)
		return x


            
            
            
        

        


