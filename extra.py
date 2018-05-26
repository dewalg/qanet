import tensorflow as tf


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.
    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def ndim(x):
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.
    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.
    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.
    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out