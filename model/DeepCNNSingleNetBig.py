import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple
# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation
    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.
    Parameters
    ----------
    input_length : int or None
        The size of the input.
    filter_size : int
        The size of the filter.
    stride : int
        The stride of the convolution operation.
    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        both borders.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.
    Returns
    -------
    int or None
        The output size corresponding to the given convolution parameters, or
        ``None`` if `input_size` is ``None``.
    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


def conv_input_length(output_length, filter_size, stride, pad=0):
    """Helper function to compute the input size of a convolution operation
    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.
    Parameters
    ----------
    output_length : int or None
        The size of the output.
    filter_size : int
        The size of the filter.
    stride : int
        The stride of the convolution operation.
    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        both borders.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.
    Returns
    -------
    int or None
        The smallest input size corresponding to the given convolution
        parameters for the given output size, or ``None`` if `output_size` is
        ``None``. For a strided convolution, any input size of up to
        ``stride - 1`` elements larger than returned will still give the same
        output size.
    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    Notes
    -----
    This can be used to compute the output size of a convolution backward pass,
    also called transposed convolution, fractionally-strided convolution or
    (wrongly) deconvolution in the literature.
    """
    if output_length is None:
        return None
    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'same':
        pad = filter_size // 2
    if not isinstance(pad, int):
        raise ValueError('Invalid pad: {0}'.format(pad))
    return (output_length - 1) * stride - 2 * pad + filter_size

class BaseConvLayer(lasagne.layers.Layer):
    """
    lasagne.layers.BaseConvLayer(incoming, num_filters, filter_size,
    stride=1, pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True,
    n=None, **kwargs)
    Convolutional layer base class
    Base class for performing an `n`-dimensional convolution on its input,
    optionally adding a bias and applying an elementwise nonlinearity. Note
    that this class cannot be used in a Lasagne network, only its subclasses
    can (e.g., :class:`Conv1DLayer`, :class:`Conv2DLayer`).
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        be a tensor of 2+`n` dimensions:
        ``(batch_size, num_input_channels, <n spatial dimensions>)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or an `n`-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or an `n`-element tuple specifying the stride of the
        convolution operation.
    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of `n` integers allows different symmetric padding
        per dimension.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If ``True``, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be an
        `n`-dimensional tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor of 2+`n` dimensions with shape
        ``(num_filters, num_input_channels, <n spatial dimensions>)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, <n spatial dimensions>)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.
    n : int or None
        The dimensionality of the convolution (i.e., the number of spatial
        dimensions of each feature map and each convolutional filter). If
        ``None``, will be inferred from the input shape.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 n=None, **kwargs):
        super(BaseConvLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n+2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.
        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        """
        Symbolically convolves `input` with ``self.W``, producing an output of
        shape ``self.output_shape``. To be implemented by subclasses.
        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`
        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        raise NotImplementedError("BaseConvLayer does not implement the "
                                  "convolve() method. You will want to "
"use a subclass such as Conv2DLayer.")
        

class TransposedConv2DLayer(BaseConvLayer):
    """
    lasagne.layers.TransposedConv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), crop=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=False, **kwargs)
    2D transposed convolution layer
    Performs the backward pass of a 2D convolution (also called transposed
    convolution, fractionally-strided convolution or deconvolution in the
    literature) on its input and optionally adds a bias and applies an
    elementwise nonlinearity.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        transposed convolution operation. For the transposed convolution, this
        gives the dilation factor for the input -- increasing it increases the
        output size.
    crop : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the transposed convolution is computed where the input and
        the filter overlap by at least one position (a full convolution). When
        ``stride=1``, this yields an output that is larger than the input by
        ``filter_size - 1``. It can be thought of as a valid convolution padded
        with zeros. The `crop` argument allows you to decrease the amount of
        this zero-padding, reducing the output size. It is the counterpart to
        the `pad` argument in a non-transposed convolution.
        A single integer results in symmetric cropping of the given size on all
        borders, a tuple of two integers allows different symmetric cropping
        per dimension.
        ``'full'`` disables zero-padding. It is is equivalent to computing the
        convolution wherever the input and the filter fully overlap.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no cropping / a full convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_input_channels, num_filters, filter_rows, filter_columns)``.
        Note that the first two dimensions are swapped compared to a
        non-transposed convolution.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: False)
        Whether to flip the filters before sliding them over the input,
        performing a convolution, or not to flip them and perform a
        correlation (this is the default). Note that this flag is inverted
        compared to a non-transposed convolution.
    output_size : int or iterable of int or symbolic tuple of ints
        The output size of the transposed convolution. Allows to specify
        which of the possible output shapes to return when stride > 1.
        If not specified, the smallest shape will be returned.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    Notes
    -----
    The transposed convolution is implemented as the backward pass of a
    corresponding non-transposed convolution. It can be thought of as dilating
    the input (by adding ``stride - 1`` zeros between adjacent input elements),
    padding it with ``filter_size - 1 - crop`` zeros, and cross-correlating it
    with the filters. See [1]_ for more background.
    Examples
    --------
    To transpose an existing convolution, with tied filter weights:
    >>> from lasagne.layers import Conv2DLayer, TransposedConv2DLayer
    >>> conv = Conv2DLayer((None, 1, 32, 32), 16, 3, stride=2, pad=2)
    >>> deconv = TransposedConv2DLayer(conv, conv.input_shape[1],
    ...         conv.filter_size, stride=conv.stride, crop=conv.pad,
    ...         W=conv.W, flip_filters=not conv.flip_filters)
    References
    ----------
    .. [1] Vincent Dumoulin, Francesco Visin (2016):
           A guide to convolution arithmetic for deep learning. arXiv.
           http://arxiv.org/abs/1603.07285,
           https://github.com/vdumoulin/conv_arithmetic
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 crop=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=False,
                 output_size=None, **kwargs):
        # output_size must be set before calling the super constructor
        if (not isinstance(output_size, T.Variable) and
                output_size is not None):
            output_size = as_tuple(output_size, 2, int)
        self.output_size = output_size
        super(TransposedConv2DLayer, self).__init__(
                incoming, num_filters, filter_size, stride, crop, untie_biases,
                W, b, nonlinearity, flip_filters, n=2, **kwargs)
        # rename self.pad to self.crop:
        self.crop = self.pad
        del self.pad

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        # first two sizes are swapped compared to a forward convolution
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        if self.output_size is not None:
            size = self.output_size
            if isinstance(self.output_size, T.Variable):
                size = (None, None)
            return input_shape[0], self.num_filters, size[0], size[1]

        # If self.output_size is not specified, return the smallest shape
        # when called from the constructor, self.crop is still called self.pad:
        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_input_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, crop)))

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.crop == 'same' else self.crop
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=self.get_W_shape(),
            subsample=self.stride, border_mode=border_mode,
            filter_flip=not self.flip_filters)
        output_size = self.output_shape[2:]
        if isinstance(self.output_size, T.Variable):
            output_size = self.output_size
        elif any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(self.W, input, output_size)
        return conved

Deconv2DLayer = TransposedConv2DLayer

class Repeat(lasagne.layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        super(Repeat, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        tensors = [input]*self.n
        stacked = theano.tensor.stack(*tensors)
        dim = [1, 0] + list(range(2, input.ndim + 1)) # suport python3
        return stacked.dimshuffle(dim)
    
class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    ds is a tuple, denotes the upsampling
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        """
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)
        """
        shp = input.shape
        upsample = T.zeros((shp[0], shp[1], shp[2] * 2, shp[3] * 2), dtype=input.dtype)
        upsample = T.set_subtensor(upsample[:, :, ::ds[0], ::ds[1]], input)
        return upsample
    
class InverseLayer(layers.MergeLayer):
    """
    The :class:`InverseLayer` class performs inverse operations
    for a single layer of a neural network by applying the
    partial derivative of the layer to be inverted with
    respect to its input: transposed layer
    for a :class:`DenseLayer`, deconvolutional layer for
    :class:`Conv2DLayer`, :class:`Conv1DLayer`; or
    an unpooling layer for :class:`MaxPool2DLayer`.
    It is specially useful for building (convolutional)
    autoencoders with tied parameters.
    Note that if the layer to be inverted contains a nonlinearity
    and/or a bias, the :class:`InverseLayer` will include the derivative
    of that in its computation.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    layer : a :class:`Layer` instance or a tuple
        The layer with respect to which the instance of the
        :class:`InverseLayer` is inverse to.
    Examples
    --------
    >>> import lasagne
    >>> from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer
    >>> from lasagne.layers import InverseLayer
    >>> l_in = InputLayer((100, 3, 28, 28))
    >>> l1 = Conv2DLayer(l_in, num_filters=16, filter_size=5)
    >>> l2 = DenseLayer(l1, num_units=20)
    >>> l_u2 = InverseLayer(l2, l2)  # backprop through l2
    >>> l_u1 = InverseLayer(l_u2, l1)  # backprop through l1
    """
    def __init__(self, incoming, layer, **kwargs):

        super(InverseLayer, self).__init__(
            [incoming, layer, layer.input_layer], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[2]

    def get_output_for(self, inputs, **kwargs):
        input, layer_out, layer_in = inputs
        return theano.grad(None, wrt=layer_in, known_grads={layer_out: input})


class DeepCNNSingleNetBig(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepCNNSingleNetBig,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(self._batch_size, self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size, self._state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Target = T.col("Target")
        self._Target.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        
        self._bottleneck_size = 64
        
        # create a small convolutional neural network
        input = lasagne.layers.InputLayer((None, self._state_length), self._State)
        self._stateInputVar = input.input_var
        actionInput = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        self._actionInputVar = actionInput.input_var
        
        taskFeatures = lasagne.layers.SliceLayer(input, indices=slice(0, self._settings['num_terrain_features']), axis=1)
        # characterFeatures = lasagne.layers.SliceLayer(network, indices=slice(-(self._state_length-self._settings['num_terrain_features']), None), axis=1)
        characterFeatures = lasagne.layers.SliceLayer(input, indices=slice(self._settings['num_terrain_features'], self._state_length), axis=1)
        print ("taskFeatures Shape:", lasagne.layers.get_output_shape(taskFeatures))
        print ("characterFeatures Shape:", lasagne.layers.get_output_shape(characterFeatures))
        print ("State length: ", self._state_length)
        
        network = lasagne.layers.ReshapeLayer(taskFeatures, (-1, 1, self._settings['num_terrain_features']))
        
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=16, filter_size=8,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        """
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        """
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._critic_task_part = network 
        
        """
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        
        network = lasagne.layers.DenseLayer(
                network, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        network = lasagne.layers.FlattenLayer(network, outdim=2)
        networkMiddle = lasagne.layers.ConcatLayer([network, characterFeatures], axis=1)
        
        networkMiddle = lasagne.layers.DenseLayer(
                networkMiddle, num_units=self._bottleneck_size,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                networkMiddle, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=16,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
                
                
        networkAct = lasagne.layers.DenseLayer(
                networkMiddle, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._actor = lasagne.layers.DenseLayer(
                    networkAct, num_units=self._action_length,
                    nonlinearity=lasagne.nonlinearities.linear)
        
        
        # networkAct = lasagne.layers.ReshapeLayer(networkAct, (-1, 99))
        # networkAct = lasagne.layers.FlattenLayer(networkAct, 2)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkMiddle))
        # networkAct = lasagne.layers.ConcatLayer([networkAct, inputLayerAction], axis=1)
        
        """
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
                
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        """
        networkActMiddle = lasagne.layers.ConcatLayer([networkMiddle, actionInput])
        
        
        networkAct = lasagne.layers.ReshapeLayer(networkActMiddle, (-1, 1, 1, self._bottleneck_size + self._action_length))
        
        networkAct = Deconv2DLayer(
            networkAct, num_filters=16, filter_size=(1,4),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        networkAct = Deconv2DLayer(
            networkAct, num_filters=32, filter_size=(1,8),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        # networkAct = lasagne.layers.ReshapeLayer(networkAct, (-1, 1, 1, 74))
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=self._settings['num_terrain_features'],
                nonlinearity=lasagne.nonlinearities.linear)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        
        networkActChar = lasagne.layers.DenseLayer(
                networkActMiddle, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        networkActChar = lasagne.layers.DenseLayer(
                networkActChar, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        networkActChar = lasagne.layers.DenseLayer(
                networkActChar, num_units=((self._state_length) - self._settings['num_terrain_features']),
                nonlinearity=lasagne.nonlinearities.linear)
        
        # networkAct = lasagne.layers.FlattenLayer(networkAct, outdim=2)
        # this should have dimensions (1,self._state_length + self._action_length)...
        ## Put the terrain features together with the character feature predictions
        networkAct = lasagne.layers.ConcatLayer([networkAct, networkActChar], axis=1) 
        """
        self._actor = lasagne.layers.DenseLayer(
                networkAct, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        """
        self._forward_dynamics_net = networkAct
    
        if (self._settings['use_stocastic_policy']):
            with_std = lasagne.layers.DenseLayer(
                    networkAct, num_units=self._action_length,
                    nonlinearity=theano.tensor.nnet.softplus)
            self._actor = lasagne.layers.ConcatLayer([self._actor, with_std], axis=1)
        # self._b_o = init_b_weights((n_out,))
        
        
        networkActReward = lasagne.layers.DenseLayer(
                networkActMiddle, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        networkActReward = lasagne.layers.DenseLayer(
                networkActReward, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        networkActReward = lasagne.layers.DenseLayer(
                networkActReward, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)        
        self._reward_net = networkActReward
        
        networkMiddle = lasagne.layers.ReshapeLayer(networkMiddle, (-1, 1, 1, self._bottleneck_size))
        
        networkActEncode = Deconv2DLayer(
            networkMiddle, num_filters=32, filter_size=(1,4),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        networkActEncode = Deconv2DLayer(
            networkActEncode, num_filters=16, filter_size=(1,8),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        networkActEncode = lasagne.layers.FlattenLayer(networkActEncode, outdim=2)
        networkActEncode = lasagne.layers.ConcatLayer([networkActEncode, characterFeatures], axis=1)
        
        networkActEncode = lasagne.layers.DenseLayer(
                networkActEncode, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
        self._encode_net = networkActEncode
        
        
          # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        
        self._target_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self._actions_shared = theano.shared(
            np.zeros((self._batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
    def setStates(self, states):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        states : a (batch_size, state_dimension) numpy array
        """
        # states = np.array(states)
        # states = np.reshape(states, (states.shape[0], 1, states.shape[1]))
        self._states_shared.set_value(states)
    def setResultStates(self, resultStates):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        resultStates : a (batch_size, state_dimension) numpy array
        """
        # resultStates = np.array(resultStates)
        # resultStates = np.reshape(resultStates, (resultStates.shape[0], 1, resultStates.shape[1]))
        self._next_states_shared.set_value(resultStates)
        
    def getEncodeNet(self):
        return self._encode_net
