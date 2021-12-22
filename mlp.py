import numpy as np

class DenseLayer:
    def __init__(
            self, 
            n_units, 
            input_size=None, 
            activation=None, 
            name=None):
        self.n_units = n_units
        self.input_size = input_size
        self.W = None
        self.name = name
        self.A = None
        self.activation = activation
        self.fn, self.df = self._select_activation_fn(activation) 

    def init_weights(self):
        np.random.seed(42)
        self.W = (np.random.randn(self.n_units, self.input_size + 1) - 0.5) * 2 / self.input_size

    def __call__(self, X):
        m_examples = X.shape[0]
        X_extended = np.hstack([np.ones((m_examples, 1)), X])
        Z = np.einsum('bi,oi->bo', X_extended, self.W)
        A = self.fn(Z)
        self.A = A
        return A
    
    def _select_activation_fn(self, activation):
        if activation == 'relu':
            fn = lambda x: np.where(x < 0, 0.0, x)
            df = lambda x: np.where(x < 0, 0.0, 1.0)
        elif activation == 'sigmoid':
            fn = lambda x: 1 / (1 + np.exp(-x))
            df = lambda x: x * (1 - x)
        elif activation == 'tanh':
            fn = lambda x: (np.exp(x) - np.exp(-1)) / (np.exp(x) + np.exp(-x))
            df = lambda x: 1 - x**2
        elif activation == 'softmax':
            fn = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
            df = lambda x: np.apply_along_axis(np.diagflat, axis=1, arr=x) - np.einsum('ij,ik->ijk', x, x)
        elif activation is None:
            fn = lambda x: x
            df = lambda x: 1.0
        else:
            NotImplementedError(f"Function {activation} cannot be used.")
        return fn, df

    
class SequentialModel:
    def __init__(self, layers, lr=0.01, momentum=0.8, loss='mse'):
        input_size = layers[0].n_units
        layers[0].init_weights()
        for layer in layers[1:]:
            layer.input_size = input_size
            input_size = layer.n_units
            layer.init_weights()
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.prev_dWs = {}
        self.loss_fns = self._select_loss_function(loss)

    def __repr__(self):
        return f"SequentialModel n_layer: {len(self.layers)}"

    def forward(self, X):
        out = self.layers[0](X)
        for layer in self.layers[1:]:
            out = layer(out)
        return out

    def _select_loss_function(self, loss):
        if loss == 'mse':
            fn = lambda y_pred, y_true: np.sum(0.5*(y_pred - y_true) ** 2) / (2*y_true.shape[0])
            df = lambda y_pred, y_true: y_pred - y_true
        else:
            raise NotImplementedError(f"Function {activation} cannot be used.")
        return fn, df
    
    def loss(self, y_pred, y_true):
        return self.loss_fns[0](y_pred, y_true)
    
    def loss_grad(self, y_pred, y_true):
        return self.loss_fns[1](y_pred, y_true)
    
    def _extend(self, vec):
        return np.hstack([np.ones((vec.shape[0], 1)), vec])

    def backprop(self, delta, layer_W, layer_a):
        da = self.layers[layer_a].df(self.layers[layer_a].A)  # the derivative of the activation fn
        if len(da.shape) > 2:
            raise NotImplementedError(f"Function {self.layers[layer_a].activation} not implemented for backprop currently.")
        return np.einsum('bz,za,ba->ba', delta, self.layers[layer_W].W[..., 1:], da)
        
    def backward(self, X, y_pred, y_true):
        # backprop through loss
        delta = self.loss_grad(y_pred, y_true)
        # backprop through last activation
        delta = np.einsum('ba,baz->bz', delta, self.layers[-1].df(self.layers[-1].A))

        dWs = {}
        # begin backprop loop. 
        # a: activated neurons before weight i
        # delta: backprop difference term of preactivated neurons after weight i
        for i in range(-1, -len(self.layers), -1):
            a = self.layers[i-1].A
            dWs[i] = np.einsum('ba,bz->za', self._extend(a), delta) / delta.shape[0]
            delta = self.backprop(delta, i, i-1)

        # final update using input X
        dWs[-len(self.layers)] = np.einsum('bi,bz->zi', self._extend(X), delta)

        # update all weights
        for k, dW in dWs.items():
            self.layers[k].W = self.layers[k].W - self.lr*(dW + self.momentum*self.prev_dWs.get(k, 0))
            
        # update previous updates for momentum term
        self.prev_dWs = dWs
        
    def train(self, X, y, iters):
        for i in range(iters):
            y_pred = self.forward(X)
            if (i%100 == 0):
                print("Iteration: ", i, "Loss: ", self.loss(y_pred, y))
            self.backward(X, y_pred, y)
    
    def evaluate(self, X, y):
        """
            this method is to evaluate our model on unseen samples
            it computes the confusion matrix and the accuracy

            X is a numpy array of shape (num_train, D) containing the testing images
                    consisting of num_train samples each of dimension D.
            y is  a numpy array of shape (num_train, D) containing the testing labels
                    consisting of num_train samples each of dimension D.
        """
        outputs = self.forward(X)
        nclasses = np.shape(y)[1]

        # 1-of-N encoding
        outputs = np.argmax(outputs, 1)
        targets = np.argmax(y, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {np.trace(cm) / np.sum(cm) * 100:0.4f}")

        return cm