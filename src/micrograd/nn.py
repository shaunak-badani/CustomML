from .engine import Value
import numpy as np

class Layer:
    """
    Defines the layer containing weights and biases
    """
    def __init__(self, nin, nout):
        self.w = Value(np.random.uniform(-1, 1, size = (nout, nin)), label = 'w')
        self.b = Value(np.random.uniform(-1, 1, size = (nout, 1)), label = 'b')
        
    def __call__(self, x):
        out_value = self.w @ x + self.b
        p = out_value.tanh()
        return p
    
    def params(self):
        return [self.w, self.b]

class MLP:
    def __init__(self, layers_array):
        num_layers = len(layers_array) - 1
        self.layers = [Layer(layers_array[i], layers_array[i + 1]) for i in range(num_layers)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params
    
    def zero_grad(self):
        for layer in self.layers:
            layer.w.grad = np.zeros_like(layer.w.grad)
            layer.b.grad = np.zeros_like(layer.b.grad)
            
    
    def train(self, data_object, epochs = 10000, output_period = 100, loss_fn = 'mse'):
        """
        Function to train the model
        args : 
        data_object : Data object which fetches batches
        Assuming that y contains raw label data
        """
        history = []
        for k in range(epochs):
            for data, target in data_object.data_loader:
                train_data, target = data_object.flatten_data(data, target)
                x = Value(train_data)
                y = target

                ypred = self.__call__(x)
                if loss_fn == 'mse':
                    # Have to fix this
                    loss = MSELossFn()
                if loss_fn == 'cross_entropy':
                    loss = ypred.softmax_with_ce(y)
                loss.backward()
                
                params = self.parameters()
                lr = 1e-3
                for param in params:
                    param.data -= lr * param.grad
                self.zero_grad()
            if k % output_period == 0:
                history.append([k, loss.data.item()])
        return np.array(history)

    def save_to_torch_model(self, model_path):
        """
        Loads the weights and biases of the model into a 
        pytorch model and saves it
        """
        from torch import nn
        import torch
        state_dict = {}
        modules = []
        for index, param in enumerate(self.parameters()):
            # odd index implies a bias
            # even index implies a weight
            if index % 2:
                state_dict[str(index // 2) + ".bias"] = \
                    torch.Tensor(param.data.squeeze())
            else:
                state_dict[str(index // 2) + ".weight"] = \
                    torch.Tensor(param.data)
                a, b = param.data.shape
                modules.append(nn.Linear(b, a))
        torch_model = nn.Sequential(*modules)
        torch_model.load_state_dict(state_dict)
        torch.save(torch_model.state_dict(), model_path)

    @staticmethod
    def load_to_model(model_path, layers):
        """
        Loads the weights of the model
        """
        import torch
        from torch import nn
        modules = []
        modules.append(nn.Linear(784, 256))
        modules.append(nn.Linear(256, 10))
        torch_model = nn.Sequential(*modules)
        torch_model.load_state_dict(torch.load(model_path))
        model = MLP(layers)
        
        for index, torch_param in \
            enumerate(torch_model.parameters()):
            # even index implies a weight
            if index % 2:
                model.layers[index // 2].b = Value(torch_param.detach().numpy().reshape(-1, 1))
            else:
                model.layers[index // 2].w = Value(torch_param.detach().numpy())
        return model
        
                

