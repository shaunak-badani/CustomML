import numpy as np

class CustomMLP:
    
    def __init__(self):
        self.w1 = np.random.normal(size = (512, 784)) * np.sqrt(1 / 784)
        self.b1 = np.random.normal(size = (512, 1))
        self.w2 = np.random.normal(size = (256, 512))* np.sqrt(1 / 512)
        self.b2 = np.random.normal(size = (256, 1))
        self.w3 = np.random.normal(size = (10, 256)) * np.sqrt(1 / 256)
        self.b3 = np.random.normal(size = (10, 1))
        self.learning_rate = 0.01
    
    def relu(self, a):
        a[a < 0] = 0
        return a
    
    def relu_der(self, a):
        j = np.empty_like(a)
        j[a < 0] = 0
        j[a > 0] = 1
        
        return j
    
    
    def sigmoid(self, a):
        j = np.empty_like(a)
        j[a < 0] = np.exp(a[a < 0]) / (1 + np.exp(a[a < 0]))
        j[a > 0] = 1 / (1 + np.exp(-a[a > 0]))
        
        return j
    
    def forward_prop(self, X, Y):
        x = X.T
        y = Y.T
                
        self.z1 = self.w1 @ x + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = self.w2 @ self.a1 + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = self.w3 @ self.a2 + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        
        p = np.exp(self.a3)
        self.p = p / p.sum(axis = 0, keepdims = True)
        self.loss = np.mean(-y * np.log(self.p))
    
    def backward_prop(self, X, Y):
        
        dJ_dw1, dJ_db1, dJ_dw2, dJ_db2, dJ_dw3, dJ_db3 = self.compute_gradients(X, Y)
        
        # Backpropagation step
        self.w3 -= self.learning_rate * dJ_dw3
        self.b3 -= self.learning_rate * dJ_db3
        
        self.w2 -= self.learning_rate * dJ_dw2
        self.b2 -= self.learning_rate * dJ_db2
        
        self.w1 -= self.learning_rate * dJ_dw1
        self.b1 -= self.learning_rate * dJ_db1
        
    def compute_gradients(self, X, Y):
        x = X.T
        y = Y.T
        
        # Backward propagation
        _, batch_size = self.p.shape
        dJ_dz3 = (self.p - y) * self.a3 * (1 - self.a3)
        dJ_dw3 = dJ_dz3 @ self.a2.T / batch_size
        dJ_db3 = np.mean(dJ_dz3, axis = 1, keepdims = True)
        
        dJ_dz2 = (self.w3.T @ dJ_dz3) * self.relu_der(self.z2)
        dJ_dw2 = dJ_dz2 @ self.a1.T / batch_size
        dJ_db2 = np.mean(dJ_dz2, axis = 1, keepdims = True)
        
        dJ_dz1 = (self.w2.T @ dJ_dz2) * self.relu_der(self.z1)
        dJ_dw1 = dJ_dz1 @ x.T / batch_size
        dJ_db1 = np.mean(dJ_dz1, axis = 1, keepdims = True)
        
        return [dJ_dw1, dJ_db1, dJ_dw2, dJ_db2, dJ_dw3, dJ_db3]
        
        
    
    def train(self, X, y, batch_size = 512, epochs = 2):
        N = X.shape[0]
        losses = []
        loss = 0
        for epoch in range(epochs):
            for ind in range(0, N, batch_size):
                self.forward_prop(X[ind:ind + batch_size], y[ind:ind + batch_size])
                self.backward_prop(X[ind:ind + batch_size], y[ind:ind + batch_size])
            self.forward_prop(X, y)
            
            print("Epoch {}: Loss {}".format(epoch, self.loss))
            losses.append(self.loss)
        return losses

    def save_model(self):
        import os
        import netCDF4 as nc
        os.system('mkdir -p ../models')
        fn = '../models/MLP1.nc'
        ds = nc.Dataset(fn, 'w', format='NETCDF4')

        layer1 = ds.createDimension('layer1', 784)
        layer2 = ds.createDimension('layer2', 512)
        layer3 = ds.createDimension('layer3', 256)
        layer4 = ds.createDimension('layer4', 10)
        singular = ds.createDimension('singular', 1)

        weights1 = ds.createVariable('w1', 'f4', ('layer2', 'layer1',))
        biases1 = ds.createVariable('b1', 'f4', ('layer2', 'singular',))
        weights1[:] = self.w1
        biases1[:] = self.b1

        weights2 = ds.createVariable('w2', 'f4', ('layer3', 'layer2',))
        biases2 = ds.createVariable('b2', 'f4', ('layer3', 'singular',))
        weights2[:] = self.w2
        biases2[:] = self.b2

        weights3 = ds.createVariable('w3', 'f4', ('layer4', 'layer3',))
        biases3 = ds.createVariable('b3', 'f4', ('layer4', 'singular',))
        weights3[:] = self.w3
        biases3[:] = self.b3

        ds.close()



