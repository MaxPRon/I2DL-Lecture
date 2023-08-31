import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return out: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        shape = x.shape
        out, cache = np.zeros(shape), np.zeros(shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Sigmoid activation function            #
        ########################################################################
        out = 1/(1+np.exp(-x))
        cache = out
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Sigmoid activation function           #
        ########################################################################
        dx = dout * cache*(1-cache)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return outputs: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        out = None
        cache = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Relu activation function               #
        ########################################################################

        relu_matrix = x > 0
        relu_matrix = relu_matrix.astype(int)
        
        out = x * relu_matrix

        cache = out
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Relu activation function              #
        ########################################################################
        relu_backward_matrix = cache > 0
        relu_backward_matrix = relu_backward_matrix.astype(int)

        dx = dout * relu_backward_matrix
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)
    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    N, M = x.shape[0], b.shape[0]
    out = np.zeros((N,M))
    ########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.    #
    # You will need to reshape the input into rows.                        #
    ########################################################################
    flattened_x = np.array([np.matrix.flatten(x[i]) for i in range(N)])
    out = flattened_x@w + b 
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the affine backward pass.                            #
    ########################################################################
    # dx = df/dx , db = df/db, dw = df/dw, 
    N = dout.shape[0]
    flattened_x = np.array([np.matrix.flatten(x[i]) for i in range(N)]) # Better Solution: np.reshape(x,(x.shape[0],-1))
    dw = (flattened_x.T).dot(dout)
    dw = flattened_x.T@dout
    
    db = np.sum(dout,axis=0)

    dx = dout@w.T
    dx = np.reshape(dx,x.shape)

    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx, dw, db