#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes predictions with a neural network trained for bird classification.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import os
import io
from argparse import ArgumentParser

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX
import lasagne

from progress import progress
import config
import data
import model
import augment


def opts_parser():
    descr = ("Computes predictions with a neural network trained for bird "
             "classification.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to load the learned weights from (.npz format)')
    parser.add_argument('outfile', metavar='OUTFILE',
            type=str,
            help='File to save the prediction curves to (.npz/.pkl format)')
    parser.add_argument('--dataset',
            type=str, default='birdclef',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--filelists',
            type=str, default='valid',
            help='Names of the filelists to use (default: %(default)s)')
    parser.add_argument('--include-side-outputs',
            action='store_true', default=False,
            help='If given and the network has multiple output layers, use '
                 'all of them, not just the first one.')
    parser.add_argument('--plot',
            action='store_true', default=False,
            help='If given, plot each spectrogram with predictions on screen.')
    parser.add_argument('--split-pool',
            action='store_true', default=False,
            help='By default, input is passed through the full network in '
                 'chunks. With this option, it is passed only up to the layer '
                 'named "before_pool", collected, joined, and passed through '
                 'remaining layers. Makes a difference except for max-pooling.')
    parser.add_argument('--saliency',
            type=int, default=None,
            help='If given, computes and saves the saliency maps with respect to '
                 'the given class.')
    parser.add_argument('--vars', metavar='FILE',
            action='append', type=str,
            default=[os.path.join(os.path.dirname(__file__), 'defaults.vars')],
            help='Reads configuration variables from a FILE of KEY=VALUE '
                 'lines. Can be given multiple times, settings from later '
                 'files overriding earlier ones. Will read defaults.vars, '
                 'then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE',
            action='append', type=str,
            help='Set the configuration variable KEY to VALUE. Overrides '
                 'settings from --vars options. Can be given multiple times.')
    return parser

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, x)
    x /= x.sum(axis=axis, keepdims=True)
    return x

def lme(x, sharpness=1, axis=-1, keepdims=False):
    ndim = x.ndim
    xmax = x.max(axis, keepdims=True)
    x = x - xmax
    if sharpness != 1:
        x *= sharpness
    np.exp(x, x)
    x = np.log(np.mean(x, axis, keepdims=keepdims))
    if sharpness != 1:
        x /= sharpness
    x += xmax if keepdims else xmax.reshape(x.shape)
    return x

def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    outfile = options.outfile
    if options.split_pool and options.saliency:
        parser.error("--split-pool and --saliency cannot be combined.")

    # read configuration files and immediate settings
    cfg = {}
    if os.path.exists(modelfile + '.vars'):
        options.vars.insert(1, modelfile + '.vars')
    for fn in options.vars:
        cfg.update(config.parse_config_file(fn))
    cfg.update(config.parse_variable_assignments(options.var))

    # read some settings into local variables
    fps = cfg['fps']
    len_min = cfg['len_min']
    len_max = cfg['len_max']

    # prepare dataset
    print("Preparing data reading...")
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)

    # - load filelists
    filelist = []
    for d in options.filelists.split(','):
        with io.open(os.path.join(datadir, 'filelists', d)) as f:
            filelist.extend(l.rstrip() for l in f if l.rstrip())

    # - create data feed
    feed, input_formats = data.prepare_datafeed(filelist, datadir, 'test', cfg)

    # - we start the generator in a background thread
    if not options.plot:
        batches = augment.generate_in_background([data.run_datafeed(feed, cfg)],
                                                 num_cached=1)
    else:
        # unless we're plotting; this would mess up the progress counter
        batches = data.run_datafeed(feed, cfg)

    print("Preparing prediction function...")
    # instantiate neural network
    input_vars = {name: T.TensorType(str(np.dtype(dtype)),
                                     (False,) * len(shape))(name)
                  for name, (dtype, shape) in input_formats.items()}
    input_shapes = {name: shape
                    for name, (dtype, shape) in input_formats.items()}
    network = model.architecture(input_vars, input_shapes, cfg)
    if isinstance(network, list) and not options.include_side_outputs:
        network = network[0]  # only use the main output

    # load saved weights
    with np.load(modelfile, encoding='latin1') as f:
        lasagne.layers.set_all_param_values(
                network, [f['param%d' % i] for i in range(len(f.files))])

    # insert guided backprop, if needed for saliency
    if options.saliency:
        from gbprop import replace_nonlinearities
        replace_nonlinearities(network, lasagne.nonlinearities.leaky_rectify)

    # create output expression(s)
    if options.split_pool:
        network_end = network
        network = next(l for l in lasagne.layers.get_all_layers(network)[::-1]
                       if l.name == 'before_pool')
    outputs = lasagne.layers.get_output(network, deterministic=True)
    if options.split_pool:
        split_input_var = T.tensor4('input2')
        split_outputs = lasagne.layers.get_output(
            network_end, {network: split_input_var}, deterministic=True)
        split_input_vars = [v for v in theano.gof.graph.inputs([split_outputs])
                            if not isinstance(v, theano.compile.SharedVariable)
                            and not isinstance(v, theano.tensor.Constant)]

    # create saliency map expression, if needed
    if options.saliency:
        saliency = theano.grad(outputs[:, options.saliency].sum(), input_vars['spect'])
        outputs = outputs + [saliency] if isinstance(outputs, list) else [outputs, saliency]

    # prepare and compile prediction function
    print("Compiling prediction function...")
    test_fn = theano.function(list(input_vars.values()), outputs,
                              on_unused_input='ignore')
    if options.split_pool:
        pool_fn = theano.function(split_input_vars, split_outputs,
                                  on_unused_input='ignore')

    # prepare plotting, if needed
    if options.plot:
        import matplotlib
        if os.environ.get('MPLBACKEND'):
            matplotlib.use(os.environ['MPLBACKEND'])  # for old versions
        import matplotlib.pyplot as plt
        with open(os.path.join(datadir, 'labels', 'labelset'), 'rb') as f:
            labelset = [l.rstrip('\r\n') for l in f]

    # run prediction loop
    print("Predicting:")
    predictions = []
    for batch in batches:
        spect = batch.pop('spect')
        if spect.shape[-2] <= len_max * fps or len_max == 0:
            # predict on full spectrogram at once
            preds = test_fn(spect=spect, **batch)
        else:
            # predict in segments of len_max, with overlap len_min
            # drop any reminder shorter than len_min (len_max if len_min == 0)
            preds = [test_fn(spect=spect[..., pos:pos + len_max * fps, :],
                     **batch)
                     for pos in range(0, (spect.shape[-2] + 1 -
                                          (len_min or len_max) * fps),
                                      (len_max - len_min) * fps)]
            if isinstance(preds[0], list):
                preds = [np.concatenate(p, axis=2 if p[0].ndim > 2 else 0)
                         for p in zip(*preds)]
            else:
                preds = np.concatenate(preds,
                                       axis=2 if preds[0].ndim > 2 else 0)
        if cfg['arch.pool'] == 'none' or '_nopool' in cfg['arch.pool']:
            if isinstance(preds, list):
                preds = [p[0, :, :, 0].T if p.ndim == 4 else p for p in preds]
            else:
                preds = preds[0, :, :, 0].T
        elif options.split_pool:
            preds = pool_fn(preds, **batch)
        predictions.append(preds)
        if options.plot:
            if spect.ndim == 4:
                spect = spect[0]  # remove batch axis
            if spect.ndim == 3:
                spect = spect[0]  # remove channel axis
            if isinstance(preds, list):
                preds, sides = preds[0], preds[1:]
            else:
                sides = []
            fig, axs = plt.subplots(2 + len(sides), 1, sharex=True)
            axs[0].imshow(np.log1p(1e-3 * spect).T[::-1], cmap='hot',
                          aspect='auto', interpolation='nearest')
            K = 5
            top_k = lme(preds, axis=0).argpartition(preds.shape[1] - 1 -
                                                    np.arange(K))[::-1][:K]
            #top_k = (preds * softmax(sides[0], axis=0).mean(axis=1, keepdims=True)).sum(axis=0).argpartition(preds.shape[1] - 1 - np.arange(K))[::-1][:K]
            #top_k = softmax(preds, axis=-1).max(axis=0).argpartition(preds.shape[1] - 1 - np.arange(K))[::-1][:K]
            #top_k[-1] = labelset.index('mphbjm')
            preds = softmax(preds, axis=-1)
            x = np.arange(len(preds)) * (len(spect) / float(len(preds)))
            for k in top_k:
                axs[1].plot(x, preds[:, k], label=labelset[k])
            #axs[1].set_ylim(0, 1.1)
            axs[1].legend(loc='best')
            for side, ax in zip(sides, axs[2:]):
                side = softmax(side, axis=0)
                ax.plot(x, side)
            plt.show()

    # save predictions
    print("Saving predictions")
    predictions = dict(zip(filelist, predictions))
    if outfile.endswith('.pkl'):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        with io.open(outfile, 'wb') as f:
            pickle.dump(predictions, f, protocol=-1)
    else:
        np.savez(outfile, **predictions)


if __name__ == "__main__":
    main()
