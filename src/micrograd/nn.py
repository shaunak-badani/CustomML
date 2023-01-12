from .engine import Value
import numpy.random as random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x): # self(x) calls this function
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
    
class MLP:
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def backward(self, lr = 0.01):
        for p in self.parameters():
            p.data -= lr * p.grad

    def train(self, x, y, epochs = 1):
        n = y.shape[0]
        print(n)

        batch_size = 32
        for k in range(epochs):
            for i in range(0, n, batch_size):
                start = i
                end = min(i + batch_size, n)
                xs = x[start:end]
                values = [self(x) for x in xs]
                print(values)
                # loss = - sum(y * values.log(values))
                # print(loss)

