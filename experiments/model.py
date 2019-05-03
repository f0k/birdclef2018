#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network architecture definition for bird classification experiment.

Author: Jan Schlüter
"""

import functools

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, Pool2DLayer, DenseLayer,
                            ExpressionLayer, dropout, batch_norm, SliceLayer,
                            GlobalPoolLayer, NonlinearityLayer, ReshapeLayer,
                            ElemwiseSumLayer, Upscale2DLayer, BiasLayer)
batch_norm_vanilla = batch_norm
try:
    from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
except ImportError:
    from lasagne.layers import BatchNormLayer


class MelBankLayer(lasagne.layers.Layer):
    """
    Creates a mel filterbank layer of `num_bands` triangular filters, with
    the first filter initialized to start at `min_freq` and the last one
    to stop at `max_freq`. Expects to process magnitude spectra created
    from samples at a sample rate of `sample_rate` with a window length of
    `frame_len` samples. Learns a vector of `num_bands + 2` values, with
    the first value giving `min_freq` in mel, and remaining values giving
    the distance to the respective next peak in mel.
    """
    def __init__(self, incoming, sample_rate, frame_len, num_bands, min_freq,
                 max_freq, trainable=True, rnd_spread=0, **kwargs):
        super(MelBankLayer, self).__init__(incoming, **kwargs)
        # mel-spaced peak frequencies
        min_mel = 1127 * np.log1p(min_freq / 700.0)
        max_mel = 1127 * np.log1p(max_freq / 700.0)
        spacing = (max_mel - min_mel) / (num_bands + 1)
        spaces = np.ones(num_bands + 2) * spacing
        spaces[0] = min_mel
        spaces = theano.shared(lasagne.utils.floatX(spaces))  # learned param
        peaks_mel = spaces.cumsum()

        # create parameter as a vector of real-valued peak bins
        peaks_hz = 700 * (T.expm1(peaks_mel / 1127))
        peaks_bin = peaks_hz * frame_len / sample_rate
        self.peaks = self.add_param(
                peaks_bin, shape=(num_bands + 2,), name='peaks',
                trainable=trainable, regularizable=False)

        # store what else is needed
        self.num_bands = num_bands
        self.rnd_spread = rnd_spread
        if rnd_spread:
            from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
            self._srng = RandomStreams(
                    lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.num_bands,)

    def get_output_for(self, input, deterministic=False, **kwargs):
        num_bins = self.input_shape[-1] or input.shape[-1]
        x = T.arange(num_bins, dtype=input.dtype).dimshuffle(0, 'x')
        peaks = self.peaks
        if not deterministic and self.rnd_spread:
            peaks *= self._srng.uniform(
                    (), 1 - self.rnd_spread, 1 + self.rnd_spread,
                    dtype=peaks.dtype)
        l, c, r = peaks[0:-2], peaks[1:-1], peaks[2:]
        # triangles are the minimum of two linear functions f(x) = a*x + b
        # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
        tri_left = (x - l) / (c - l)
        # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
        tri_right = (x - r) / (c - r)
        # combine by taking the minimum of the left and right sides
        tri = T.minimum(tri_left, tri_right)
        # and clip to only keep positive values
        bank = T.maximum(0, tri)

        # the dot product of the input with this filter bank is the output
        return T.dot(input, bank)


class PowLayer(lasagne.layers.Layer):
    def __init__(self, incoming, exponent=lasagne.init.Constant(0), **kwargs):
        super(PowLayer, self).__init__(incoming, **kwargs)
        self.exponent = self.add_param(exponent, shape=(), name='exponent', regularizable=False)

    def get_output_for(self, input, **kwargs):
        return input ** self.exponent


class PCENLayer(lasagne.layers.Layer):
    def __init__(self, incoming,
                 log_s=lasagne.init.Constant(np.log(0.025)),
                 log_alpha=lasagne.init.Constant(0),
                 log_delta=lasagne.init.Constant(0),
                 log_r=lasagne.init.Constant(0),
                 eps=1e-6, **kwargs):
        super(PCENLayer, self).__init__(incoming, **kwargs)
        num_bands = self.input_shape[-1]
        self.log_s = self.add_param(log_s, shape=(num_bands,),
                                    name='log_s', regularizable=False)
        self.log_alpha = self.add_param(log_alpha, shape=(num_bands,),
                                        name='log_alpha', regularizable=False)
        self.log_delta = self.add_param(log_delta, shape=(num_bands,),
                                        name='log_delta', regularizable=False)
        self.log_r = self.add_param(log_r, shape=(num_bands,),
                                    name='log_r', regularizable=False)
        self.eps = eps

    def get_output_for(self, input, **kwargs):
        def smooth_step(current_in, previous_out, s):
            one = T.constant(1)
            return [(one - s) * previous_out + s * current_in]
        init = input[:, :, 0]  # start smoother from first frame
        s = T.exp(self.log_s).dimshuffle('x', 'x', 0)
        smoother = theano.scan(fn=smooth_step,
                               sequences=[input.transpose(2, 0, 1, 3)],
                               non_sequences=[s],
                               outputs_info=[init],
                               strict=True)[0].transpose(1, 2, 0, 3)
        alpha = T.exp(self.log_alpha)
        delta = T.exp(self.log_delta)
        r = T.exp(self.log_r)
        return (input / (self.eps + smoother)**alpha + delta)**r - delta**r


def logmeanexp(x, axis=None, keepdims=False, sharpness=5):
    # in between maximum (high sharpness) and mean (low sharpness)
    # https://arxiv.org/abs/1411.6228, Eq. 6
    # return T.log(T.mean(T.exp(sharpness * x), axis, keepdims=keepdims)) / sharpness
    # more stable version (Theano can only stabilize the plain logsumexp)
    xmax = T.max(x, axis, keepdims=True)
    xmax2 = T.max(x, axis, keepdims=keepdims)
    x = sharpness * (x - xmax)
    y = T.log(T.mean(T.exp(x), axis, keepdims=keepdims))
    return y / sharpness + xmax2


def logmeanexp_pool2d(layer, pool_size, sharpness=5):
    xmax = Pool2DLayer(layer, pool_size, mode='max')
    xmax2 = Upscale2DLayer(xmax, pool_size)
    layer = ElemwiseSumLayer([layer, xmax2], coeffs=[sharpness, -sharpness],
                             cropping=[None, None, 'lower', 'lower'])
    layer = NonlinearityLayer(layer, T.exp)
    layer = Pool2DLayer(layer, pool_size, mode='average_exc_pad')
    layer = NonlinearityLayer(layer, T.log)
    layer = ElemwiseSumLayer([layer, xmax], coeffs=[1. / sharpness, 1])
    return layer


def noisy_or(x, axis=None, keepdims=False):
    one = T.constant(1, x.dtype)
    return one - T.exp(T.sum(T.log(one - x), axis=axis, keepdims=keepdims))


def reeu(x, alpha=1):
    return T.switch(x > 0, alpha * T.expm1(x), 0)


def grouped_dropout(layer, groups, rate=0.5):
    ndim = len(layer.output_shape)
    # turn groups into explicit axis
    layer = ReshapeLayer(layer, ([0], groups, -1) + tuple([i] for i in range(2, ndim)))
    # apply dropout, sharing the mask after the groups axis
    layer = dropout(layer, rate, shared_axes=range(2, ndim + 1))
    # reshape back
    layer = ReshapeLayer(layer, ([0], -1) + tuple([i] for i in range(3, ndim + 1)))
    return layer


def spatial_logsoftmax(x, axis=1):
    x = x - T.max(x, axis, keepdims=True)
    return x - T.log(T.sum(T.exp(x), axis, keepdims=True))


def spatial_softmax(x, axis=1):
    x = T.exp(x - T.max(x, axis, keepdims=True))
    return x / T.sum(x, axis, keepdims=True)


def log_categorical_crossentropy(preds, truth, axis=-1):
    if truth.ndim == preds.ndim:
        return -T.sum(truth * preds, axis=axis)
    else:
        assert preds.ndim == 2
        assert truth.ndim == 1
        assert axis in (1, -1)
        return -preds[T.arange(truth.shape[0]), truth]


class AttentionPoolLayer(lasagne.layers.MergeLayer):
    def get_output_shape_for(self, input_shapes):
        output_shape = list(input_shapes[0])
        return (output_shape[0], output_shape[1])

    def get_output_for(self, inputs, **kwargs):
        data, att = inputs
        # apply softmax over temporal axis
        att = spatial_softmax(att, axis=2)
        # take mean over channel axis for multi-attention
        att = T.mean(att, axis=1, keepdims=True)
        # multiply and sum over temporal axis
        data = (data * att).sum(axis=2)
        # drop remaining spatial dimension, only leave channels
        return data[:, :, 0]


class AddUpLayer(lasagne.layers.MergeLayer):
    def __init__(self, layer1, layer2):
        super(AddUpLayer, self).__init__([layer1, layer2])

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        x, y = inputs
        if self.input_shapes[0][1] > self.input_shapes[1][1]:
            return T.inc_subtensor(x[:, :y.shape[1], :y.shape[2], :y.shape[3]], y)
        else:
            return x + y


def residual_unit(layer, num_filters, filter_size, stride, Winit=None,
                  batchnorm=True, shortcut_mode='project', preact=True,
                  postact=False, dilate=False):
    # Residual unit following http://arxiv.org/abs/1512.03385 for the shortcuts
    # and http://arxiv.org/abs/1603.05027 (Figure 1 b) for the residual units.
    # Note: We have a bias in front of every nonlinearity in the residual path,
    # and a bias after every change of the tensor layout in the shortcut path.
    # Any additional biases in the shortcut path would be redundant.
    kwargs = {} if Winit is None else {'W': Winit}
    kwargs.update(nonlinearity=None)

    def residual_act(layer):
        if batchnorm:
            layer = BatchNormLayer(layer)
        else:
            layer = BiasLayer(layer)
        return NonlinearityLayer(layer,
                                 nonlinearity=lasagne.nonlinearities.rectify)

    # Shortcut: 1x1 convolution or avg-pooling if layout changes...
    input_shape = layer.output_shape
    if input_shape[1] != num_filters:
        if shortcut_mode == 'project' or num_filters < input_shape[1]:
            shortcut = Conv2DLayer(layer, num_filters, 1, stride, **kwargs)
        elif stride == 1 or stride == (1, 1) or stride == [1, 1]:
            shortcut = layer
        elif shortcut_mode == 'pool':
            shortcut = Pool2DLayer(layer, pool_size=stride,
                                   mode='average_exc_pad')
        elif shortcut_mode == 'slice':
            shortcut = layer
            for d, s in enumerate(lasagne.utils.as_tuple(stride, 2)):
                shortcut = SliceLayer(shortcut, slice(None, None, s),
                                      axis=2 + d)
    elif stride != 1 or isinstance(stride, (tuple, list)) and max(stride) > 1:
        raise NotImplementedError(
            "Error: The shortcut connection for same number of filters, "
            "but lower resolution is not defined in literature. You may "
            "want to use average pooling, it's just not implemented yet.")
    # ...direct copy otherwise
    else:
        shortcut = layer

    # Residual: batchnorm > relu > conv > batchnorm > relu > conv
    # (unless preact=False: in this case we skip the first batchnorm > relu)
    kwargs.update(b=None)
    if preact:
        layer = residual_act(layer)
    layer = Conv2DLayer(layer, num_filters, filter_size, stride, pad='same',
                        **kwargs)
    layer = residual_act(layer)
    if not dilate:
        layer = Conv2DLayer(layer, num_filters, filter_size, 1, pad='same',
                            **kwargs)
    else:
        # TODO: add padding to DilatedConv2DLayer
        layer = lasagne.layers.PadLayer(
                layer, tuple(((np.array(filter_size) - 1) * 2 + 1) // 2))
        layer = lasagne.layers.DilatedConv2DLayer(
                layer, num_filters, filter_size, dilation=2, **kwargs)

    # Return sum of the residual and shortcut
    # (unless postact=True: in this case we add another batchnorm > relu)
    layer = AddUpLayer(layer, shortcut)
    if postact:
        layer = residual_act(layer)
    return layer


def residual_block(layer, num_units, num_filters, filter_size, stride, Winit=None, batchnorm=True, shortcut_mode='project', preact=True, postact=False, dilate=False, act_bug=False):
    for u in range(num_units):
        layer = residual_unit(layer, num_filters, filter_size,
                              stride if u > 0 else 1, Winit, batchnorm,
                              shortcut_mode,
                              preact=(not act_bug and (preact or u > 0) or
                                      act_bug and preact and u == 0),
                              postact=postact and u == num_units - 1,
                              dilate=dilate)
    return layer


def parse_arch_def(layer, arch, cfg, cfg_base='arch'):
    """
    Common code for turning arch* and arch_meta* into a stack of layers.
    """
    # common configuration
    kwargs = dict(W=lasagne.init.Orthogonal())
    if cfg[cfg_base + '.nonlin'] == 'lrelu':
        kwargs['nonlinearity'] = lasagne.nonlinearities.leaky_rectify
    elif cfg[cfg_base + '.nonlin'] == 'relu':
        kwargs['nonlinearity'] = lasagne.nonlinearities.rectify
    elif cfg[cfg_base + '.nonlin'] == 'elu':
        kwargs['nonlinearity'] = lasagne.nonlinearities.elu
    elif cfg[cfg_base + '.nonlin'] == 'reeu':
        kwargs['nonlinearity'] = reeu
    else:
        raise ValueError("Unknown %s.nonlin=%s" %
                         (cfg_base, cfg[cfg_base + '.nonlin']))
    maybe_batch_norm = (batch_norm if cfg[cfg_base + '.batch_norm']
                        else lambda x: x)
    convdrop_rate = cfg.get(cfg_base + '.convdrop_rate', 0.1)
    if cfg.get(cfg_base + '.convdrop', 'none') == 'none':
        maybe_dropout = lambda x: x
    elif cfg[cfg_base + '.convdrop'] == 'independent':
        maybe_dropout = lambda x: dropout(x, convdrop_rate)
    elif cfg[cfg_base + '.convdrop'] == 'channels':
        maybe_dropout = lambda x: dropout(x, convdrop_rate, shared_axes=(2, 3))
    elif cfg[cfg_base + '.convdrop'] == 'bands':
        maybe_dropout = lambda x: dropout(x, convdrop_rate, shared_axes=(1, 2))
    elif cfg[cfg_base + '.convdrop'] == 'groups':
        maybe_dropout = lambda x: grouped_dropout(
                x, cfg[cfg_base + '.conv_groups'], convdrop_rate)
        print("Warning: %s.convdrop=groups does not make sense. Add a single "
              "groupdrop layer to the architecture definition instead." %
              cfg_base)
    else:
        raise ValueError("Unknown %s.convdrop=%s" %
                         (cfg_base, cfg[cfg_base + '.convdrop']))

    # layers
    for layeridx, layerdef in enumerate(arch.split(',')):
        if ':' in layerdef:
            kind, shape = layerdef.split(':', 1)
        else:
            kind, shape = layerdef, ''
        if ':' in shape:
            shape, extra = shape.split(':')
        else:
            extra = ''
        try:
            shape = list(map(int, shape.split('x')))
        except ValueError:
            pass
        if kind in ('conv', 'Conv', 'convg1'):
            num_filters = shape[0]
            filter_size = shape[1:3]
            stride = shape[3:] or 1
            if filter_size[0] < 0:
                filter_size[0] += layer.output_shape[2] + 1
            elif filter_size[1] < 0:
                filter_size[1] += layer.output_shape[3] + 1
            if kind == 'Conv':
                layer = dropout(layer, cfg.get(cfg_base + '.fulldrop', 0.5))
                groups = cfg[cfg_base + '.dense_groups']
            elif kind in ('conv', 'convg1'):
                if layeridx > 0:
                    layer = maybe_dropout(layer)
                    if kind == 'convg1':
                        groups = 1
                    else:
                        groups = cfg[cfg_base + '.conv_groups']
                else:
                    groups = 1
            if groups > 1:
                kwargs['num_groups'] = groups
            else:
                kwargs.pop('num_groups', None)
            layer = Conv2DLayer(layer, num_filters, filter_size,
                                stride=stride, **kwargs)
            if layeridx == 0 and cfg.get(cfg_base + '.firstconv_meanfree') == 'params':
                layer.W = layer.W - T.mean(layer.W, axis=(2, 3),
                                           keepdims=True)
            layer = maybe_batch_norm(layer)
        elif kind in ('pool', 'avgpool'):
            if shape[0] < 0:
                shape[0] += layer.output_shape[2] + 1
            if shape[1] < 0:
                shape[1] += layer.output_shape[3] + 1
            stride = shape[2:] or None
            shape = shape[:2]
            mode = 'max' if kind == 'pool' else 'average_exc_pad'
            layer = Pool2DLayer(layer, shape, stride, mode=mode)
        elif kind.startswith('lme') and kind.endswith('pool'):
            sharpness = float(kind[3:-4])
            layer = logmeanexp_pool2d(layer, shape, sharpness=sharpness)
        elif kind == 'groupdrop':
            layer = grouped_dropout(layer, shape[0], float(extra))
        elif kind.startswith('resblock'):
            num_units, num_filters = shape[:2]
            filter_size = shape[2:4]
            stride = shape[4:6] or 1
            is_start = 'start' in kind or 'begin' in kind
            is_end = 'end' in kind
            is_dilated = 'dilated' in kind
            shortcut_mode = cfg.get('resnet.shortcut', 'project')
            layer = residual_block(
                    layer, num_units, num_filters, filter_size,
                    stride, Winit=kwargs['W'],
                    batchnorm=cfg[cfg_base + '.batch_norm'],
                    shortcut_mode=shortcut_mode, preact=not is_start,
                    postact=is_end, dilate=is_dilated,
                    act_bug=cfg.get('resnet.act_bug', False))
        elif kind in ('dense', 'full'):
            if layeridx > 0:
                layer = dropout(layer, cfg.get(cfg_base + '.fulldrop', 0.5))
            layer = DenseLayer(layer, shape[0], **kwargs)
            layer = maybe_batch_norm(layer)
        else:
            raise ValueError('Unknown layer type "%s"' % kind)
    return layer


def architecture(input_vars, input_shapes, cfg):
    spect = input_vars['spect']
    spect_shape = input_shapes['spect']
    if len(spect_shape) == 3:
        # insert channels dimension
        spect = spect.dimshuffle(0, 'x', 1, 2)
        spect_shape = spect_shape[:1] + (1,) + spect_shape[1:]
    layer = InputLayer(spect_shape, input_var=spect)

    # filterbank, if any
    if cfg['filterbank'] == 'mel':
        import audio
        filterbank = audio.create_mel_filterbank(
                cfg['sample_rate'], cfg['frame_len'], cfg['mel_bands'],
                cfg['mel_min'], cfg['mel_max'])
        filterbank = filterbank[:spect_shape[-1]].astype(theano.config.floatX)
        filterbank /= float(2**15 - 1)  # counter wav file sample scale
        layer = DenseLayer(
                layer, num_units=cfg['mel_bands'],
                num_leading_axes=-1, W=T.constant(filterbank), b=None,
                nonlinearity=None)
    elif cfg['filterbank'] == 'mel_learn':
        layer = MelBankLayer(
                layer, cfg['sample_rate'], cfg['frame_len'],
                cfg['mel_bands'], cfg['mel_min'], cfg['mel_max'],
                rnd_spread=cfg['arch.mel_rndspread'],
                trainable=cfg['arch.mel_trainable'])
    elif cfg['filterbank'] != 'none':
        raise ValueError("Unknown filterbank=%s" % cfg['filterbank'])

    # magnitude transformation, if any
    if cfg['magscale'] == 'log':
        layer = ExpressionLayer(layer, lambda x: T.log(T.maximum(1e-7, x)))
    elif cfg['magscale'] == 'log1p':
        layer = ExpressionLayer(layer, T.log1p)
    elif cfg['magscale'].startswith('log1p_learn'):
        # learnable log(1 + e^a * x), with given initial a (or default 0)
        a = float(cfg['magscale'][len('log1p_learn'):] or 0)
        a = T.exp(theano.shared(lasagne.utils.floatX(a)))
        layer = lasagne.layers.ScaleLayer(layer, scales=a,
                                          shared_axes=(0, 1, 2, 3))
        layer = ExpressionLayer(layer, T.log1p)
    elif cfg['magscale'].startswith('pow_learn'):
        # learnable x^sigmoid(a), with given initial a (or default 0)
        a = float(cfg['magscale'][len('pow_learn:'):] or 0)
        a = T.nnet.sigmoid(theano.shared(lasagne.utils.floatX(a)))
        layer = PowLayer(layer, exponent=a)
    elif cfg['magscale'] == 'pcen':
        layer = PCENLayer(layer)
        if cfg.get('pcen_fix_alpha'):
            layer.params[layer.log_alpha].remove("trainable")
    elif cfg['magscale'] != 'none':
        raise ValueError("Unknown magscale=%s" % cfg['magscale'])

    # standardization per frequency band
    layer = batch_norm_vanilla(layer, axes=(0, 2), beta=None, gamma=None)

    # freely configurable convolutional neural network
    arch = cfg.get('arch', None)
    if arch is None or arch == 'ismir2016':
        arch = 'conv:32x3x3,conv:32x3x3,pool:3x3,conv:32x3x3,conv:32x3x3,conv:64x3x-Shift,pool:3xShift,Conv:Fullx9x-1,Conv:Fullx1x1'
    if cfg.get('arch.nolastfreqpool'):
        arch = arch.replace('pool:3xShift', 'pool:3x1')
    arch = arch.replace('Shift', str(cfg.get('arch.shiftable', 3)))
    arch = arch.replace('Full', str(cfg.get('arch.fullsize', 256)))
    if arch != '':
        layer = parse_arch_def(layer, arch, cfg)
    layer = dropout(layer, cfg.get('arch.lastdrop', cfg.get('arch.fulldrop', 0.5)))

    # head with temporal pooling and classification
    if cfg['arch.pool'].startswith('attention'):
        num_attention = int(cfg['arch.pool'].split(':', 1)[1])
    else:
        num_attention = 0
    layer = Conv2DLayer(layer, cfg['classes'] + num_attention, 1,
                        nonlinearity=None,
                        W=lasagne.init.Orthogonal())
    layer.name = 'before_pool'  # marker for predict.py --split-pool
    if num_attention:
        # with attention, the last `num_attention` channels denote the attention
        # we computed them along with the class logits for efficiency
        layer_att = lasagne.layers.SliceLayer(
                layer, slice(-num_attention, None), axis=1)
        layer = lasagne.layers.SliceLayer(
                layer, slice(None, -num_attention), axis=1)
    if cfg['arch.prepool'] == 'sigmoid':
        layer = NonlinearityLayer(layer, lasagne.nonlinearities.sigmoid)
    elif cfg['arch.prepool'] == 'softmax':
        layer = NonlinearityLayer(layer, spatial_softmax)
    elif cfg['arch.prepool'] == 'logsoftmax':
        layer = NonlinearityLayer(layer, spatial_logsoftmax)
    elif cfg['arch.prepool'] == 'none':
        pass
    else:
        raise ValueError("Unknown arch.prepool=%s" % cfg['arch.prepool'])
    if cfg['arch.pool'] == 'max':
        layer = GlobalPoolLayer(layer, T.max, name='globalpool')
    elif cfg['arch.pool'] == 'mean':
        layer = GlobalPoolLayer(layer, T.mean, name='globalpool')
    elif cfg['arch.pool'].startswith('logmeanexp:'):
        a = float(cfg['arch.pool'].split(':', 1)[1])
        layer = GlobalPoolLayer(layer,
                                functools.partial(logmeanexp, sharpness=a),
                                name='globalpool')
    elif cfg['arch.pool'] == 'noisy-or':
        layer = GlobalPoolLayer(layer, noisy_or, name='globalpool')
    elif cfg['arch.pool'].startswith('attention:'):
        layer = AttentionPoolLayer([layer, layer_att])
    elif cfg['arch.pool'].startswith('attention_nopool:'):
        pass  # computes attentions but does not pool
    elif cfg['arch.pool'] == 'none':
        pass
    else:
        raise ValueError("Unknown arch.pool=%s" % cfg['arch.pool'])

    # separate network foot processing metadata
    if cfg['arch_meta']:
        # input
        mlayer = InputLayer(input_shapes['meta'], input_vars['meta'])
        # configurable part
        mlayer = parse_arch_def(mlayer, cfg['arch_meta'], cfg, 'arch_meta')
        # head with dense layer, but no nonlinearity yet
        mlayer = DenseLayer(mlayer, num_units=cfg['classes'],
                            nonlinearity=None,
                            W=lasagne.init.Orthogonal())
        # combine with spectrogram-processing CNN
        if cfg['arch'] == '':
            layer = mlayer
        else:
            layer = ElemwiseSumLayer((layer, mlayer),
                                     coeffs=[1, cfg.get('arch_meta.weight', 1)])

    # output nonlinearity
    if cfg['arch.output'] == 'sigmoid':
        outlayer = [NonlinearityLayer(layer, lasagne.nonlinearities.sigmoid)]
    elif cfg['arch.output'] == 'softmax':
        outlayer = [NonlinearityLayer(layer, lasagne.nonlinearities.softmax)]
    elif cfg['arch.output'] == 'linear':
        outlayer = [layer]
    else:
        raise ValueError("Unknown arch.output=%s" % cfg['arch.output'])
    if cfg.get('arch.output_bg') == 'sigmoid':
        outlayer.append(NonlinearityLayer(layer, lasagne.nonlinearities.sigmoid))
    if cfg.get('arch.pool', '').startswith('attention_nopool:'):
        outlayer.append(layer_att)
    return outlayer[0] if len(outlayer) == 1 else outlayer


def cost(preds, target_vars, designation, cfg):
    cost_spec = cfg['cost']
    preds_bg = preds.pop(1) if cfg.get('arch.output_bg') else None
    attention = preds.pop(1) if cfg.get('arch.pool', '').startswith('attention_nopool:') else None
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if designation == 'train' and 'mixed_label' in target_vars:
        targets = target_vars['mixed_label']
    else:
        targets = target_vars['label']
    if designation == 'train' and 'pseudo_label' in target_vars:
        if targets.ndim == 1:
            targets = lasagne.utils.one_hot(targets, cfg['classes'])
        # adding the labels is equivalent to adding two losses
        targets += target_vars['pseudo_label']
    if designation == 'train' and cfg.get('mix_fg_bg'):
        if targets.ndim == 1:
            targets = lasagne.utils.one_hot(targets, cfg['classes'])
        targets += cfg['mix_fg_bg'] * target_vars['label_bg']
    if cost_spec == 'softmax':
        if cfg.get('pred_clip'):
            preds = T.clip(preds, cfg['pred_clip'], 1 - cfg['pred_clip'])
        cost = lasagne.objectives.categorical_crossentropy(preds, targets)
    elif cost_spec == 'logsoftmax':
        cost = log_categorical_crossentropy(preds, targets)
    elif cost_spec == 'attention_logsoftmax':
        # computes a loss per timestep, then averages losses with attention
        attention = spatial_softmax(attention, axis=(2, 3))
        attention = T.mean(attention, axis=1)
        if targets.ndim == 1:
            targets = lasagne.utils.one_hot(targets, cfg['classes'])
        cost = log_categorical_crossentropy(preds,
                                            targets.dimshuffle(0, 1, 'x', 'x'),
                                            axis=1)
        cost = (cost * attention).sum(axis=(1, 2))
    elif cost_spec.startswith('ce'):
        # all predictions should be zero except for the true target
        if targets.ndim == 1:
            targets = lasagne.utils.one_hot(targets, cfg['classes'])
        if cost_spec.startswith('ce_bgtopk:'):
            # ignore the top k predictions (assumed valid background species)
            k = int(cost_spec.split(':')[1])
            kth_pred = T.topk(preds, k, axis=-1, sorted=False).min(axis=-1, keepdims=True)
            mask = (preds < kth_pred)
            # the true target must not be ignored
            mask = T.maximum(mask, targets)
        else:
            mask = 1
        # compare with binary cross-entropy
        if cfg.get('pred_clip'):
            preds = T.clip(preds, cfg['pred_clip'], 1 - cfg['pred_clip'])
        cost = lasagne.objectives.binary_crossentropy(preds, targets) * mask
    elif cost_spec == 'none':
        cost = 0  # only useful if cost_bg is given
    else:
        raise ValueError("Unknown cost=%s" % cost_spec)
    if preds_bg is not None:
        cost_spec = cfg['cost_bg']
        if cost_spec == 'ce':
            cost += lasagne.objectives.binary_crossentropy(
                    preds_bg, target_vars['label_bg']).mean(axis=-1)
        else:
            raise ValueError("Unknown cost_bg=%s" % cost_spec)
    # weight examples according to given weights
    if designation == 'train' and 'weight' in target_vars:
        cost *= T.shape_padright(target_vars['weight'], cost.ndim - 1)
    return cost
