#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trains a neural network for bird classification.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser
try:
    import cPickle as pickle
except ImportError:
    import pickle

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
    descr = "Trains a neural network for bird classification."
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to save the learned weights to (.npz format)')
    parser.add_argument('--dataset',
            type=str, default='birdclef',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--validate',
            action='store_true', default=False,
            help='Monitor validation loss (disabled by default)')
    parser.add_argument('--no-validate',
            action='store_false', dest='validate',
            help='Disable monitoring validation loss (disabled by default)')
    parser.add_argument('--save-errors',
            action='store_true', default=False,
            help='If given, save error log in {MODELFILE%%.npz}.err.npz.')
    parser.add_argument('--keep-state',
            action='store_true', default=False,
            help='If given, save the complete training state after each epoch '
                 'in {MODELFILE%%.npz}.state, and load it to continue from '
                 'there if the script is restarted.')
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


def get_state(network, updates):
    return ([p.get_value() for p in updates.keys()] +
            lasagne.layers.get_all_param_values(network, trainable=False))


def restore_state(network, updates, state):
    for p, s in zip(updates.keys(), state):
        p.set_value(s)
    lasagne.layers.set_all_param_values(
            network, state[len(updates):], trainable=False)


def save_model(modelfile, network, cfg):
    np.savez(modelfile, **{'param%d' % i: p for i, p in enumerate(
            lasagne.layers.get_all_param_values(network))})
    write = 'wb' if sys.version_info[0] == 2 else 'w'
    with io.open(modelfile + '.vars', write) as f:
        f.writelines('%s=%s\n' % kv for kv in cfg.items())


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile

    # read configuration files and immediate settings
    cfg = {}
    for fn in options.vars:
        cfg.update(config.parse_config_file(fn))
    cfg.update(config.parse_variable_assignments(options.var))

    # prepare dataset
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)

    print("Preparing training data feed...")
    with io.open(os.path.join(datadir, 'filelists', 'train')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]
    train_feed, train_formats = data.prepare_datafeed(filelist, datadir,
                                                      'train', cfg)

    # If told so, we plot some mini-batches on screen.
    if cfg.get('plot_datafeed'):
        import matplotlib.pyplot as plt
        for batch in data.run_datafeed(train_feed, cfg):
            plt.matshow(np.log(batch['spect'][0]).T, aspect='auto',
                        origin='lower', cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(str(batch['label'][0]))
            plt.show()

    # We start the mini-batch generator and augmenter in one or more
    # background threads or processes (unless disabled).
    bg_threads = cfg['bg_threads']
    bg_processes = cfg['bg_processes']
    if not bg_threads and not bg_processes:
        # no background processing: just create a single generator
        batches = data.run_datafeed(train_feed, cfg)
    elif bg_threads:
        # multithreading: create a separate generator per thread
        batches = augment.generate_in_background(
                [data.run_datafeed(feed, cfg)
                 for feed in data.split_datafeed(train_feed, bg_threads, cfg)],
                num_cached=bg_threads * 2)
    elif bg_processes:
        # multiprocessing: single generator is forked along with processes
        batches = augment.generate_in_background(
                [data.run_datafeed(train_feed, cfg)] * bg_processes,
                num_cached=bg_processes * 25,
                in_processes=True)

    # If told so, we benchmark the creation of a given number of mini-batches.
    if cfg.get('benchmark_datafeed'):
        print("Benchmark: %d mini-batches of %d items " %
              (cfg['benchmark_datafeed'], cfg['batchsize']), end='')
        if bg_threads:
            print("(in %d threads): " % bg_threads)
        elif bg_processes:
            print("(in %d processes): " % bg_processes)
        else:
            print("(in main thread): ")
        import time
        import itertools
        t0 = time.time()
        next(itertools.islice(batches, cfg['benchmark_datafeed'],
                              cfg['benchmark_datafeed']), None)
        t1 = time.time()
        print (t1 - t0)
        return

    # - prepare validation data generator
    if options.validate:
        print("Preparing validation data feed...")
        with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
            filelist_val = [l.rstrip() for l in f if l.rstrip()]
        val_feed, val_formats = data.prepare_datafeed(filelist_val, datadir,
                                                      'valid', cfg)
        if bg_threads or bg_processes:
            multi = bg_threads or bg_processes
            val_feed = data.split_datafeed(val_feed, multi, cfg)

        def run_val_datafeed():
            if bg_threads or bg_processes:
                return augment.generate_in_background(
                        [data.run_datafeed(feed, cfg)
                         for feed in val_feed],
                        num_cached=multi, in_processes=bool(bg_processes))
            else:
                return data.run_datafeed(val_feed, cfg)

    print("Preparing training function...")
    # instantiate neural network
    input_vars = {name: T.TensorType(str(np.dtype(dtype)),
                                     (False,) * len(shape))(name)
                  for name, (dtype, shape) in train_formats.items()}
    input_shapes = {name: shape
                    for name, (dtype, shape) in train_formats.items()}
    network = model.architecture(input_vars, input_shapes, cfg)
    print("- %d layers (%d with weights), %f mio params" %
          (len(lasagne.layers.get_all_layers(network)),
           sum(hasattr(l, 'W') for l in lasagne.layers.get_all_layers(network)),
           lasagne.layers.count_params(network, trainable=True) / 1e6))
    print("- weight shapes: %r" % [
            l.W.get_value().shape
            for l in lasagne.layers.get_all_layers(network)
            if hasattr(l, 'W') and hasattr(l.W, 'get_value')])
    cost_vars = dict(input_vars)

    # prepare for born-again-network, if needed
    if cfg.get('ban'):
        network2 = model.architecture(input_vars, input_shapes, cfg)
        with np.load(cfg['ban'], encoding='latin1') as f:
            lasagne.layers.set_all_param_values(
                    network2, [f['param%d' % i] for i in range(len(f.files))])
        cost_vars['pseudo_label'] = lasagne.layers.get_output(
                network2, deterministic=True)

    # load pre-trained weights, if needed
    if cfg.get('init_from'):
        param_values = []
        for fn in cfg['init_from'].split(':'):
            with np.load(fn, encoding='latin1') as f:
                param_values.extend(f['param%d' % i]
                                    for i in range(len(f.files)))
        lasagne.layers.set_all_param_values(network, param_values)
        del param_values

    # create cost expression
    outputs = lasagne.layers.get_output(network, deterministic=False)
    cost = T.mean(model.cost(outputs, cost_vars, 'train', cfg))
    if cfg.get('l2_decay', 0):
        cost_l2 = lasagne.regularization.regularize_network_params(
                network, lasagne.regularization.l2) * cfg['l2_decay']
    else:
        cost_l2 = 0

    # prepare and compile training function
    params = lasagne.layers.get_all_params(network, trainable=True)
    initial_eta = cfg['initial_eta']
    eta_decay = cfg['eta_decay']
    eta_decay_every = cfg.get('eta_decay_every', 1)
    eta_cycle = tuple(map(float, str(cfg['eta_cycle']).split(':')))
    if eta_cycle == (0,):
        eta_cycle = (1,)  # so eta_cycle=0 equals disabling it
    patience = cfg.get('patience', 0)
    trials_of_patience = cfg.get('trials_of_patience', 1)
    patience_criterion = cfg.get('patience_criterion',
                                 'valid_loss' if options.validate
                                 else 'train_loss')
    momentum = cfg['momentum']
    first_params = params[:cfg['first_params']]
    first_params_eta_scale = cfg['first_params_eta_scale']
    if cfg['learn_scheme'] == 'nesterov':
        learn_scheme = lasagne.updates.nesterov_momentum
    elif cfg['learn_scheme'] == 'momentum':
        learn_scheme = lasagne.update.momentum
    elif cfg['learn_scheme'] == 'adam':
        learn_scheme = lasagne.updates.adam
    else:
        raise ValueError('Unknown learn_scheme=%s' % cfg['learn_scheme'])
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    if not first_params or first_params_eta_scale == 1:
        updates = learn_scheme(cost + cost_l2, params, eta, momentum)
    else:
        grads = theano.grad(cost + cost_l2, params)
        updates = learn_scheme(grads[len(first_params):],
                               params[len(first_params):], eta, momentum)
        if first_params_eta_scale > 0:
            updates.update(
                    learn_scheme(grads[:len(first_params)], first_params,
                                 eta * first_params_eta_scale, momentum))
    print("Compiling training function...")
    train_fn = theano.function(list(input_vars.values()), cost, updates=updates,
                               on_unused_input='ignore')

    # prepare and compile validation function, if requested
    if options.validate:
        print("Compiling validation function...")
        outputs_test = lasagne.layers.get_output(network, deterministic=True)
        cost_test = T.mean(model.cost(outputs_test, input_vars, 'valid', cfg))
        if isinstance(outputs_test, (list, tuple)):
            outputs_test = outputs_test[0]
        val_fn = theano.function([input_vars[k] for k in val_formats],
                                 [cost_test, outputs_test],
                                 on_unused_input='ignore')

    # restore previous training state, or create fresh training state
    state = {}
    if options.keep_state:
        statefile = modelfile[:-len('.npz')] + '.state'
        if os.path.exists(statefile):
            print("Restoring training state...")
            state = np.load(modelfile[:-len('.npz')] + '.state',
                            encoding='latin1')
            restore_state(network, updates, state['network'])
    epochs = cfg['epochs']
    epochsize = cfg['epochsize']
    batches = iter(batches)
    if options.save_errors:
        errors = state.get('errors', [])
    if first_params and cfg['first_params_log']:
        first_params_hist = []
        if options.keep_state and os.path.exists(modelfile[:-4] + '.hist.npz'):
            with np.load(modelfile[:-4] + '.hist.npz') as f:
                first_params_hist = list(zip(*(f['param%d' % i]
                                             for i in range(len(first_params)))))
    if patience > 0:
        best_error = state.get('best_error', np.inf)
        best_state = state.get('best_state') or get_state(network, updates)
        patience = state.get('patience', patience)
        trials_of_patience = state.get('trials_of_patience', trials_of_patience)
    epoch = state.get('epoch', 0)
    del state

    # run training loop
    print("Training:")
    for epoch in range(epoch, epochs):
        # actual training
        err = 0
        for batch in progress(
                range(epochsize), min_delay=.5,
                desc='Epoch %d/%d: Batch ' % (epoch + 1, epochs)):
            err += train_fn(**next(batches))
            if not np.isfinite(err):
                print("\nEncountered NaN loss in training. Aborting.")
                sys.exit(1)
            if first_params and cfg['first_params_log'] and (batch % cfg['first_params_log'] == 0):
                first_params_hist.append(tuple(param.get_value()
                                               for param in first_params))
                np.savez(modelfile[:-4] + '.hist.npz',
                         **{'param%d' % i: param
                            for i, param in enumerate(zip(*first_params_hist))})

        # report training loss
        print("Train loss: %.3f" % (err / epochsize))
        if options.save_errors:
            errors.append(err / epochsize)

        # compute and report validation loss, if requested
        if options.validate:
            import time
            t0 = time.time()
            # predict in mini-batches
            val_err = 0
            val_batches = 0
            preds = []
            truth = []
            for batch in run_val_datafeed():
                e, p = val_fn(**batch)
                val_err += np.sum(e)
                val_batches += 1
                preds.append(p)
                truth.append(batch['label'])
            t1 = time.time()
            # join mini-batches
            preds = np.concatenate(preds) if len(preds) > 1 else preds[0]
            truth = np.concatenate(truth) if len(truth) > 1 else truth[0]
            # show results
            print("Validation loss: %.3f" % (val_err / val_batches))
            from eval import evaluate
            results = evaluate(preds, truth)
            print("Validation error: %.3f" % (1 - results['accuracy']))
            print("Validation MAP: %.3f" % results['map'])
            print("(took %.2f seconds)" % (t1 - t0))
            if options.save_errors:
                errors.append(val_err / val_batches)
                errors.append(1 - results['accuracy'])
                errors.append(results['map'])

        # update learning rate and/or apply early stopping, if needed
        if patience > 0:
            if patience_criterion == 'train_loss':
                cur_error = err / epochsize
            elif patience_criterion == 'valid_loss':
                cur_error = val_err / val_batches
            elif patience_criterion == 'valid_error':
                cur_error = 1 - results['accuracy']
            elif patience_criterion == 'valid_map':
                cur_error = 1 - results['map']
            if cur_error <= best_error:
                best_error = cur_error
                best_state = get_state(network, updates)
                patience = cfg['patience']
            else:
                patience -= 1
                if patience == 0:
                    if eta_decay_every == 'trial_of_patience' and eta_decay != 1:
                        eta.set_value(eta.get_value() * lasagne.utils.floatX(eta_decay))
                    restore_state(network, updates, best_state)
                    patience = cfg['patience']
                    trials_of_patience -= 1
                    print("Lost patience (%d remaining trials)." % trials_of_patience)
                    if trials_of_patience == 0:
                        break
        if eta_decay_every != 'trial_of_patience' and eta_decay != 1 and \
                (epoch + 1) % eta_decay_every == 0:
            eta.set_value(eta.get_value() * lasagne.utils.floatX(eta_decay))
        if eta_cycle[epoch % len(eta_cycle)] != 1:
            eta.set_value(eta.get_value() *
                          lasagne.utils.floatX(eta_cycle[epoch % len(eta_cycle)]))

        # store current training state, if needed
        if options.keep_state:
            state = {}
            state['epoch'] = epoch + 1
            state['network'] = get_state(network, updates)
            if options.save_errors:
                state['errors'] = errors
            if patience > 0:
                state['best_error'] = best_error
                state['best_state'] = best_state
                state['patience'] = patience
                state['trials_of_patience'] = trials_of_patience
            with open(statefile, 'wb') as f:
                pickle.dump(state, f, -1)
            del state

        # for debugging: print memory use and break into debugger
        #import resource, psutil
        #print("Memory usage: %.3f MiB / %.3f MiB" %
        #      (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
        #       psutil.Process().memory_info()[0] / float(1024**2)))
        #import pdb; pdb.set_trace()

    # save final network
    print("Saving final model")
    save_model(modelfile, network, cfg)
    if options.save_errors:
        np.savez(modelfile[:-len('.npz')] + '.err.npz',
                 np.asarray(errors).reshape(epoch + 1, -1))


if __name__ == "__main__":
    main()
