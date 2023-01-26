import math
import numpy as np

class Value:
    """
    Base class which overrides standard mathematical operations
    so that gradient functions are computed and stored while
    the operation performs
    """
    
    def __init__(self, data, _children = (), _op='', label = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
        self.grad = np.zeros_like(self.data, dtype = np.float64)
    
    def __add__(self, other):
        if(isinstance(other, (int, float))):
            other = Value(np.array(other))

        a = self.data.shape[0]
        b = 1
        if(len(self.data.shape) > 1):
            b = self.data.shape[1]
        c = other.data.shape[0]
        d = 1
        if(len(other.data.shape) > 1):
            d = other.data.shape[1]
        
        assert ((a == c) or (b == d)), "Can't add these matrices"
        out = Value(self.data + other.data, (self, other), '+')
        if a == c:
            if b != d:
                assert d == 1 or b == 1, "Can't broadcast these matrices"
        
        if d == b:
            if c != a:
                assert c == 1 or b == 1, "Can't broadcast these matrices"
        
        
        def _backward():
            other_grad = out.grad
            self_grad = out.grad
            if(a == c):
                if b != d:
                    if d == 1:
                        other_grad = np.sum(out.grad, axis = 1, keepdims = True)
                    else:
                        self_grad = np.sum(out.grad, axis = 1, keepdims = True)

            if(d == b):
                if c != a:
                    if c == 1:
                        other_grad = np.sum(out.grad, axis = 0, keepdims = True)
                    else:
                        self_grad = np.sum(out.grad, axis = 0, keepdims = True)
                    
            self.grad += self_grad
            other.grad += other_grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            other = Value(other)
        return self + (-other)
    
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(np.array([[other]]))
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            grad_add = other.data * out.grad
            if self.data.shape[1] == 1:
                grad_add = np.sum(grad_add, axis = 1, keepdims = True)
            
            if self.data.shape[0] == 1:
                grad_add = np.sum(grad_add, axis = 0, keepdims = True)
                
            self.grad += grad_add
            
            grad_add = self.data * out.grad
            if other.data.shape[1] == 1:
                grad_add = np.sum(grad_add, axis = 1, keepdims = True)
            
            if other.data.shape[0] == 1:
                grad_add = np.sum(grad_add, axis = 0, keepdims = True)
            
            other.grad += grad_add
        out._backward = _backward
        return out
    
    def __neg__(self):
        return (self * -1) 
    
    def __matmul__(self, other): # self @ other == w @ x
        a, b = self.data.shape
        c, d = other.data.shape
        assert (b == c) or (a == d), "Can't multiply matrices"
        if b != c:
            return other @ self
        
        out = Value(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        out = Value(self.data**other, (self, ), f"**{other}")
        
        def _backward():
            self.grad += other * self.data**(other - 1)
        out._backward = _backward
        return out
    
    
    def exp(self):
        out = Value(np.exp(self.data), (self, ), f"exp({self})")
        
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    
    def log(self):
        out = Value(np.log(self.data), (self, ), f"log({self})")
        
        def _backward():
            self.grad += out.grad / out.data
        out._backward = _backward
        return out
    
    
    def tanh(self):
        out = Value(np.tanh(self.data), (self, ), 'tanh')
        
        def _backward():
            self.grad += out.grad * (1 - out.data**2)
        
        out._backward = _backward
        return out
    
    def sum(self):
        out = Value(np.sum(self.data, keepdims = True), (self, ), 'sigma')
        
        def _backward():
            # for each element of the gradient array,
            # broadcast and add the out grad
            self.grad += out.grad
        out._backward = _backward
        return out
    
    def max(self, axis = None):
        """
        This max function works but we don't use it for softmax
        """
        if axis is None:
            raise NotImplementedError("Not supported")
        else:                
                
            out = Value(np.max(self.data, axis = axis, keepdims = True), (self, ), f"max along axis = {axis}")
            
            def _backward():
                grad = np.zeros(self.data.shape)
                max_indices = np.argmax(self.data, axis = axis)
                range_along_indices = np.arange(max_indices.size)
                if axis == 0:
                    grad[max_indices, range_along_indices] = out.grad
                else:
                    grad[range_along_indices, max_indices] = out.grad
                self.grad += grad
            out._backward = _backward
        return out
    
    def softmax_with_ce(self, ylabels):
        softmax = np.exp(self.data - self.data.max())
        softmax /= softmax.sum(axis = 0)
        
        num_classes, num_samples = self.data.shape
        one_hot = np.zeros((num_classes, num_samples))
        one_hot[ylabels.astype('int'), np.arange(num_samples)] = 1.0
        loss = -np.sum(one_hot * np.log(softmax), keepdims = True)
        out = Value(loss, (self, ), "Softmax with CE")
        
        def _backward():
            grad = softmax - one_hot
            self.grad += grad
            
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()