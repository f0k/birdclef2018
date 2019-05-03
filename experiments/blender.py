#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Finds optimal ensemble weights for a collection of predictions.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser

import numpy as np

from eval import load_labels, evaluate, softmax


def opts_parser():
    descr = "Finds optimal ensemble weights for a collection of predictions."
    parser = ArgumentParser(description=descr)
    parser.add_argument('infile', nargs='+', metavar='INFILE',
            type=str,
            help='File to load the prediction curves from (.npz/.pkl format). '
                 'If given multiple times, prediction curves will be averaged. '
                 'If ending in ":VALUE", will initially weight by VALUE.')
    parser.add_argument('--init-weights',
            type=float, default=1.0,
            help='Initial weights for models that are not explicitly given a '
                 'weight (default: %(default)s)')
    parser.add_argument('--dataset',
            type=str, default='birdclef',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--labelfile',
            type=str, default='fg.tsv',
            help='Name of the label file to use (default: %(default)s)')
    parser.add_argument('--labelfile-background',
            type=str, default=None,
            help='Optional: Name of the background label file to use '
                 '(default: %(default)s)')
    parser.add_argument('--filelist',
            type=str, default='valid',
            help='Name of the file list to use (default: %(default)s)')
    parser.add_argument('--softmax',
            action='store_true', default=False,
            help='If given, apply a softmax to the predictions (before '
                 'ensembling).')
    parser.add_argument('--criterion',
            type=str, default='fg+bg', choices=('fg', 'bg', 'fg+bg'),
            help='Score to optimize for: MAP of the foreground species (fg), '
                 'background species (bg) or their sum (fg+bg). Ignored '
                 'if --labelfile-background is not given. '
                 '(default: %(default)s)')
    parser.add_argument('--strategy',
            type=str, default='greedy', choices=('greedy', 'hyperopt',
                                                 'hyperopt_zero',
                                                 'hyperopt_log'),
            help='Which search strategy to use: greedy, hyperopt, '
                 'hyperopt_zero (a variant explicitly dropping models), '
                 'or hyperopt_log (a variant sampling weights logarithmically)')
    parser.add_argument('--seed',
            type=int, default=None,
            help='Optionally fix the random seed to use for the greedy search.')
    parser.add_argument('--patience',
            type=int, default=None,
            help='Control the number of evaluations per step (greedy strategy) '
                 'or in total (hyperopt strategy). (default: 10 for greedy, '
                 '1000 for hyperopt)')
    return parser


def print_results(results):
    # cannot directly call sys.stdout.isatty() due to bug in hyperopt
    # (https://github.com/hyperopt/hyperopt/pull/496)
    if getattr(sys.stdout, "isatty", lambda: False)():
        BOLD = '\033[1m'
        UNBOLD = '\033[0m'
    else:
        BOLD = UNBOLD = ''
    print(("MAP: %s%.3f%s, " % (BOLD, results['map'], UNBOLD)) +
          (("MAP-bg: %.3f, " % results['map-bg'])
           if 'map-bg' in results else '') +
          ", ".join("top-%d: %.2f" % (i + 1, v)
                    for i, v in enumerate(results['top-acc'])))


def summarize(results, criterion):
    if criterion == 'fg':
        return results['map']
    elif criterion == 'bg':
        return results['map-bg']
    elif criterion == 'fg+bg':
        return results['map'] + results['map-bg']
    else:
        raise ValueError('Unknown criterion "%s"' % criterion)


def scipy_target(w, preds, weights, k, truth, truth_bg, criterion,
                 return_results=False):
    weights = weights.copy()
    weights[k] = w
    results = evaluate((preds * weights[:, np.newaxis, np.newaxis]).sum(axis=0),
                       truth, truth_bg)
    if return_results:
        return summarize(results, criterion), results
    else:
        return -summarize(results, criterion)


def tune_weight(preds, weights, k, truth, truth_bg, criterion, patience):
    # search for the optimal value for weights[k] (in-place)
    args = (preds, weights, k, truth, truth_bg, criterion)
    if not np.any(weights) or list(np.nonzero(weights)[0]) == [k]:
        # shortcut if k is the only active model
        weights[k] = 1
        return scipy_target(1, *args, return_results=True)
    import scipy.optimize
    w = weights[k]
    opt = scipy.optimize.bracket(scipy_target, 0, w or weights.max(), args,
                                 maxiter=patience)
    w = opt[1]
    opt = scipy.optimize.fmin(scipy_target, w, args, ftol=0.0005,
                              maxfun=patience, disp=False)
    weights[k] = opt[0]
    return scipy_target(weights[k], *args, return_results=True)


def tune_weights(preds, weights, truth, truth_bg=None, criterion='fg+bg',
                 patience=10):
    # greedily tune contribution weights in random order until converged
    assert len(preds) == len(weights)
    K = len(weights)
    weights = weights.copy()
    untried = set(range(K))
    best_score = 0
    while untried:
        # pick random weight to modify
        k = np.random.choice(list(untried))
        # optimize it
        new_score, results = tune_weight(preds, weights, k, truth, truth_bg,
                                         criterion, patience)
        # compare to current best
        if new_score > best_score + 0.0001:
            best_score = new_score
            untried = set(range(K)) - {k}
            print(weights)
            print_results(results)
        else:
            untried -= {k}
    return weights


def hyperopt_target(preds, weights, truth, truth_bg, criterion):
    weights = np.asarray(weights)
    preds = (preds * weights[:, np.newaxis, np.newaxis]).sum(axis=0)
    results = evaluate(preds, truth, truth_bg)
    score = summarize(results, criterion)
    if score > hyperopt_target.best_score:
        hyperopt_target.best_score = score
        print(weights)
        print_results(results)
    return -score


def tune_with_hyperopt(preds, truth, truth_bg=None, criterion='fg+bg',
                       patience=1000, explicit_zeros=False, log_scale=False):
    from hyperopt import fmin, tpe, hp
    if log_scale:
        sampler = lambda name: hp.loguniform(name, -6, 0)
    else:
        sampler = lambda name: hp.uniform(name, 0, 1)
    if explicit_zeros:
        space = [hp.choice('m%d' % (k + 1),
                           [0, sampler('w%d' % (k + 1))])
                 for k in range(len(preds))]
    else:
        space = [sampler('w%d' % (k + 1)) for k in range(len(preds))]
    hyperopt_target.best_score = 0
    best = fmin(
            fn=lambda weights: hyperopt_target(preds, weights, truth, truth_bg,
                                               criterion),
            space=space, algo=tpe.suggest, max_evals=patience
            )
    weights = [best.get('w%d' % (k + 1), 0) for k in range(len(preds))]
    return np.asarray(weights)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    infiles = options.infile
    if not options.labelfile_background:
        options.criterion = 'fg'

    # load file lists
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)
    with io.open(os.path.join(datadir, 'filelists', options.filelist)) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]

    # load network predictions
    if options.softmax:
        preprocess = lambda x: softmax(x.max(axis=0) if x.ndim == 2 else x)
    else:
        preprocess = lambda x: x.max(axis=0) if x.ndim == 2 else x
    weights = [float(infile.rsplit(':', 1)[1])
               if ':' in infile else options.init_weights
               for infile in infiles]
    infiles = [infile.rsplit(':', 1)[0] for infile in infiles]
    preds = []
    failed = set()  # to skip non-existing file names; that was convenient
    for k, infile in enumerate(infiles):
        try:
            p = np.load(infile, encoding='latin1')
        except Exception:
            print('Could not read "%s". Skipping.' % infile)
            failed.add(k)
        else:
            preds.append([preprocess(p[fn]).ravel() for fn in filelist])
    weights = [w for k, w in enumerate(weights) if k not in failed]
    infiles = [fn for k, fn in enumerate(infiles) if k not in failed]
    weights = np.array(weights)
    preds = np.stack(preds)

    # load labels
    truth = load_labels(os.path.join(datadir, 'labels', options.labelfile))
    truth = np.array([truth[fn] for fn in filelist])
    if options.labelfile_background:
        truth_bg = load_labels(os.path.join(datadir, 'labels',
                                            options.labelfile_background),
                               multilabel=True)
        truth_bg = [truth_bg[fn] for fn in filelist]
        truth_bg_onehot = np.zeros(preds.shape[1:], dtype=np.bool)
        for row, fg, bg in zip(truth_bg_onehot, truth, truth_bg):
            row[fg] = True
            row[bg] = True

    # tune weights
    if options.seed is not None:
        np.random.seed(options.seed)
    if options.strategy == 'greedy':
        weights = tune_weights(
                preds, weights, truth,
                truth_bg if options.labelfile_background else None,
                options.criterion, patience=options.patience or 10)
    elif options.strategy in ('hyperopt', 'hyperopt_zero', 'hyperopt_log'):
        weights = tune_with_hyperopt(
                preds, truth,
                truth_bg if options.labelfile_background else None,
                options.criterion, patience=options.patience or 1000,
                explicit_zeros=(options.strategy == 'hyperopt_zero'),
                log_scale=(options.strategy == 'hyperopt_log'))

    # print results
    for fn, weight in zip(infiles, weights):
        if weight > 0:
            print("%s:%g \\" % (fn, weight))


if __name__ == "__main__":
    main()
