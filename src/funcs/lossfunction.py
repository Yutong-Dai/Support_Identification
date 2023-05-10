import numpy as np
from scipy import sparse
from numpy.linalg import eigvalsh
from numba import jit


class LogisticLoss:
    def __init__(self, X, y, datasetName=None, weight_decay=0.0):
        self.n, self.p = X.shape
        self.X, self.y = X, y
        self.X_batch, self.y_batch = None, None
        self.expterm_batch = None
        self.expterm = None
        self.sigmoid_batch = None
        self.sigmoid = None
        self.datasetName = datasetName
        self.weight_decay = weight_decay

    def __str__(self):
        info = ""
        if self.datasetName is not None:
            info += "Dataset:{:.>48}\n".format(self.datasetName)
        info += "Data Size:{:.>38}n={}, p={}\nLoss Function:{:.>34}Logistic\n".format(
            '', self.n, self.p, '')
        return info

    def _set_expterm(self, weight, bias=0, idx=None):
        if idx is not None:
            self.y_batch = self.y[idx, :]
            self.X_batch = self.X[idx, :]
            self.expterm_batch = np.exp(-(self.y_batch)
                                        * (self.X_batch @ weight + bias))
        else:
            self.expterm = np.exp(-(self.y) * (self.X @ weight + bias))

    def func(self, weight):
        """
        # Don't delete this function
        function value of logistic loss function evaluate at the given point weight.
        f(weight) = frac{1}{n} sum_{i=1}^n log(1+exp(y_i * weight^T * x_i))
        """
        f = self.evaluate_function_value(weight, bias=0, idx=None)
        return f

    def evaluate_function_value(self, weight, bias=0, idx=None):
        """
        function value of logistic loss function evaluate at the given point (weight, bias).
        f(weight,bias) = frac{1}{n} sum_{i=1}^n log(1+exp(y_i * weight^T * x_i))
        """
        if self.weight_decay > 0:
            perturb = 0.5 * self.weight_decay * (weight.T@weight).item()
        else:
            perturb = 0.0
        self._set_expterm(weight, bias, idx)
        if idx is not None:
            f = np.sum(np.log(1 + self.expterm_batch)) / len(idx)
        else:
            f = np.sum(np.log(1 + self.expterm)) / self.n
        return f + perturb

    def _set_sigmoid(self, idx=None):
        if idx is not None:
            self.sigmoid_batch = 1 - (1 / (1 + self.expterm_batch))
        else:
            self.sigmoid = 1 - (1 / (1 + self.expterm))

    def gradient(self, xk, idx=None, return_table=False):
        """
        need to be called after `evaluate_function_value` to get correct `expterm`
        """
        self._set_sigmoid(idx)
        if return_table:
            if idx is not None:
                table = (-self.y_batch * self.sigmoid_batch) * self.X_batch.toarray()
            else:
                try:
                    table = (-self.sigmoid * self.y) * self.X.toarray()
                except Exception as e:
                    print(str(e))
                    return None, None
            table = table.T
            gradient = np.mean(table, axis=1, keepdims=True)
            return gradient + self.weight_decay * xk, table + self.weight_decay * xk
        else:
            if idx is not None:
                gradient = -((self.sigmoid_batch * self.y_batch).T @
                             self.X_batch) / len(idx)
            else:
                gradient = -((self.sigmoid * self.y).T @ self.X) / self.n

            return gradient.T + self.weight_decay * xk



class LeastSquares:
    def __init__(self, X, y, datasetName=None, weight_decay=0.0):
        self.n, self.p = X.shape
        self.X, self.y = X, y
        self.datasetName = datasetName
        self.weight_decay = weight_decay

    def __str__(self):
        info = ""
        if self.datasetName is not None:
            info += "Dataset:{:.>48}\n".format(self.datasetName)
        info += "Data Size:{:.>38}n={}, p={}\nLoss Function:{:.>34}LeastSquares\n".format(
            '', self.n, self.p, '')
        return info

    
    def func(self, weight):
        return self.evaluate_function_value(weight, idx=None)

    def evaluate_function_value(self, weight, bias=0, idx=None):
        self.weight = weight
        if self.weight_decay > 0:
            perturb = 0.5 * self.weight_decay * (weight.T@weight).item()
        else:
            perturb = 0.0

        if idx is None:
            self.matvec = self.X @ weight - self.y
            f = 0.5 * np.sum(self.matvec * self.matvec) / self.n
            
        else:
            self.X_batch = self.X[idx, :]
            self.y_batch = self.y[idx, :]
            self.matvec =  self.X_batch @ weight - self.y_batch
            f = 0.5 * np.sum(self.matvec * self.matvec) / len(idx)
        return f + perturb


    def gradient(self, xk, idx=None, return_table=False):
        """
        need to be called after `evaluate_function_value` to get correct `expterm`
        """
        if return_table:
            if idx is None:
                try:
                    table = self.matvec * self.X.toarray()
                except Exception as e:
                    print(str(e))
                    return None, None
            else:
                table = self.matvec * self.X_batch.toarray()
            table = table.T
            gradient = np.mean(table, axis=1, keepdims=True)
            return gradient + self.weight_decay * xk, table + self.weight_decay * xk
        else:
            if idx is None:
                gradient = self.matvec.T @ self.X / self.n
            else:
                gradient = self.matvec.T @ self.X_batch / len(idx)
            return gradient.T + self.weight_decay * self.weight

if __name__ == "__main__":
    import scipy
    from scipy.sparse import csr_matrix, random
    import numpy as np
    X = scipy.sparse.random(10, 5, density=0.9, format='csr', random_state=0)
    np.random.seed(0)
    beta = np.random.randn(5, 1)
    y = np.random.randn(10, 1)
    weight_decay = 1e-5
    f = LeastSquares(X, y, weight_decay=weight_decay)
    
    f_batched = f.evaluate_function_value(beta, idx=[0,1,2,3,4,5,6,7,8,9])
    grad_batched = f.gradient(beta, idx=[0,1,2,3,4,5,6,7,8,9])
    f_full = f.func(beta)
    grad_full = f.gradient(beta)
    assert f_batched == f_full, "batched and full should be the same"
    assert np.allclose(grad_batched, grad_full), "batched and full gradient should be the same"

    f_batched = f.evaluate_function_value(beta, idx=[1,7])
    grad_batched = f.gradient(beta, idx=[1,7])

    f_answer = (1/(2*len([1,7]))) * ( (np.sum(X[[1], :]@beta) - y[1])**2 +  (np.sum(X[[7], :]@beta) - y[7])**2 ) + 0.5 * weight_decay * (beta.T@beta).item()
    X_batch = X[[1,7], :]
    # grad_answer = X_batch.T@(X_batch@beta - y[[1,7], :]) / len([1,7])
    grad_answer = (((X[[1], :]@beta - y[1]).item() * X[[1], :].T).toarray() + ((X[[7], :]@beta - y[7]).item() * X[[7], :].T).toarray()) / len([1,7]) + weight_decay * beta
    assert f_batched == f_answer, f"f_batched:{f_batched} and f_answer:{f_answer} should be the same"
    assert np.allclose(grad_batched, grad_answer), "grad_batched and grad_answer should be the same"

    f_batched = f.evaluate_function_value(beta, idx=[1,7])
    grad_batched, table = f.gradient(beta, idx=[1,7], return_table=True)
    table_answer = np.hstack( ( ((X[[1], :]@beta - y[1]).item() * X[[1], :].T).toarray(), ((X[[7], :]@beta - y[7]).item() * X[[7], :].T).toarray() ) ) + weight_decay * beta
    assert np.allclose(table, table_answer), "table and table_answer should be the same"
    print("test passed")