#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluates bird species predictions against ground truth.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser

import numpy as np


def opts_parser():
    descr = "Evaluates bird species predictions against ground truth."
    parser = ArgumentParser(description=descr)
    parser.add_argument('infile', nargs='+', metavar='INFILE',
            type=str,
            help='File to load the prediction curves from (.npz/.pkl format). '
                 'If given multiple times, prediction curves will be '
                 'averaged. If ending in ":VALUE", will weight by VALUE.')
    parser.add_argument('--dataset',
            type=str, default='birdclef',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--labelfile',
            type=str, default='fg.tsv',
            help='Name of the label file to use (default: %(default)s)')
    parser.add_argument('--labelfile-background',
            type=str, default=None,
            help='Optional: Name of the background label file to use '
                 ' (default: %(default)s)')
    parser.add_argument('--filelist',
            type=str, default='valid',
            help='Name of the file list to use (default: %(default)s)')
    parser.add_argument('--softmax',
            action='store_true', default=False,
            help='If given, apply a softmax to the predictions (before '
                 'ensembling).')
    parser.add_argument('--no-maxpool',
            action='store_false', dest='maxpool', default=True,
            help='If given, do not apply temporal max pooling to the '
                 ' predictions (before ensembling).')
    parser.add_argument('--save-predictions',
            type=str, default=None,
            help='If given, save ensembled predictions to the given .npz/.pkl '
                 'file instead of evaluating them.')
    parser.add_argument('--birdclef-filter',
            action='store_true', default=False,
            help='If given, filter foreground species by birdclef year.')
    parser.add_argument('--subset', metavar='FRACTION',
            type=float, default=1,
            help='Optionally evaluate on a random fraction of examples.')
    parser.add_argument('--subset-class', metavar='FRACTION',
            type=float, default=1,
            help='Optionally evaluate on a random fraction of classes.')
    parser.add_argument('--subset-seed',
            type=int, default=None,
            help='Random seed to use for --subset or --subset-class.')
    return parser


def load_labels(labelfile, multilabel=False):
    with io.open(labelfile) as f:
        labels = dict(l.rstrip('\r\n').split('\t', 1) for l in f)
    if not multilabel:
        return {fn: int(label) for fn, label in labels.items()}
    else:
        return {fn: list(map(int, label.split(','))) if label else []
                for fn, label in labels.items()}


def evaluate(predictions, truth, truth_bg=None, top_k=5):
    assert len(predictions) == len(truth)
    predictions = np.asarray(predictions)
    truth = np.asarray(truth)
    if truth_bg is not None:
        assert len(predictions) == len(truth_bg)
        if getattr(truth_bg, 'shape', None) == predictions.shape:
            truth_bg_onehot = truth_bg
        else:
            truth_bg_onehot = np.zeros(predictions.shape, dtype=np.bool)
            for row, fg, bg in zip(truth_bg_onehot, truth, truth_bg):
                row[fg] = True
                row[bg] = True

    top_predictions = np.argsort(predictions, axis=-1)[:, ::-1]
    top_correct = (top_predictions == truth[:, np.newaxis])

    # MAP-score: same as mean reciprocal rank for single target class
    ranks = top_correct.argmax(axis=-1)  # rank of correct class in prediction
    map_score = (1. / (1 + ranks)).mean()  # mean reciprocal rank

    # top-k accuracy for 1...top_k
    top_accuracies = np.cumsum(top_correct[:, :top_k], axis=-1).mean(axis=0)

    # MAP-score for background species
    if truth_bg is not None:
        N, C = predictions.shape
        # top_correct_bg[n,k] tells if the kth top species for file n is correct
        # to create it, we index into truth_bg_onehot using the prediction order
        top_correct_bg = truth_bg_onehot[np.arange(N)[:, np.newaxis],
                                         top_predictions]
        # we can use this to compute the precision at k for every file and k
        prec_at_k = (np.cumsum(top_correct_bg, axis=-1) /
                     np.arange(1, C + 1, dtype=np.double))
        # and the average precision (averaged over all ground truth species)
        avg_prec = ((prec_at_k * top_correct_bg).sum(axis=-1) /
                    top_correct_bg.sum(axis=-1))
        # finally yielding the mean average precision
        map_score_bg = avg_prec.mean()

    result = {
            'map': map_score,
            'top-acc': top_accuracies,
            'accuracy': top_accuracies[0],
             }
    if truth_bg is not None:
        result['map-bg'] = map_score_bg
    return result


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    infiles = options.infile

    # load file lists
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)
    with io.open(os.path.join(datadir, 'filelists', options.filelist)) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]

    # use a random subset, if needed
    if options.subset_seed is not None:
        np.random.seed(options.subset_seed)
    if options.subset < 1:
        subset = np.random.choice(len(filelist),
                                  size=int(options.subset * len(filelist)),
                                  replace=False)
        subset = set(subset)
        filelist = [fn for i, fn in enumerate(filelist) if i in subset]

    # load network predictions
    if options.softmax and options.maxpool:
        preprocess = lambda x: softmax(x.max(axis=0) if x.ndim == 2 else x)
    elif options.maxpool:
        preprocess = lambda x: x.max(axis=0) if x.ndim == 2 else x
    elif options.softmax:
        preprocess = softmax
    else:
        preprocess = lambda x: x
    weights = [float(infile.rsplit(':', 1)[1]) if ':' in infile else 1
               for infile in infiles]
    infiles = [infile.rsplit(':', 1)[0] for infile in infiles]
    preds = np.load(infiles[0], encoding='latin1')
    preds = {fn: preprocess(preds[fn]) * weights[0] for fn in filelist}
    if len(infiles) > 1:
        preds = {fn: preds[fn] / len(infiles) for fn in filelist}
        for infile, weight in zip(infiles[1:], weights[1:]):
            morepreds = np.load(infile, encoding='latin1')
            for fn in preds:
                preds[fn] += preprocess(morepreds[fn]) * weight / len(infiles)
        del morepreds

    # optional birdclef filter
    if options.birdclef_filter:
        with io.open(os.path.join(datadir, 'labels', 'labelset')) as f:
            labelset = [l.rstrip('\r\n') for l in f]
        fg_mask = {}
        for y in '2014', '2015', '2017':
            with io.open(os.path.join(datadir, 'labels', 'labelset_' + y)) as f:
                labelsubset = set(l.rstrip('\r\n') for l in f)
            fg_mask[y] = np.array([l not in labelsubset for l in labelset])
        preds = {fn: preds[fn] - 1e100 * fg_mask[fn.rsplit('CLEF', 1)[1][:4]]
                 for fn in filelist}

    # optionally save ensembled predictions instead of evaluating
    if options.save_predictions:
        if options.save_predictions.endswith('.pkl'):
            try:
                import cPickle as pickle
            except ImportError:
                import pickle
            with io.open(options.save_predictions, 'wb') as f:
                pickle.dump(preds, f, protocol=-1)
        else:
            np.savez(options.save_predictions, **preds)
        return

    # load labels
    truth = load_labels(os.path.join(datadir, 'labels', options.labelfile))
    if options.labelfile_background:
        truth_bg = load_labels(os.path.join(datadir, 'labels',
                                            options.labelfile_background),
                               multilabel=True)

    # use a random subset of classes, if needed
    if options.subset_class < 1:
        num_classes = next(iter(preds.values())).shape[0]
        class_set = np.random.choice(num_classes,
                                     int(options.subset_class * num_classes),
                                     replace=False)
        class_set = set(class_set)
        filelist = [fn for fn in filelist if truth[fn] in class_set]
        fg_mask = np.array([l not in class_set for l in range(num_classes)])
        preds = {fn: preds[fn] - 1e100 * fg_mask for fn in filelist}

    # evaluate
    results = evaluate(
                [preds[fn].ravel() for fn in filelist],
                [truth[fn] for fn in filelist],
                [truth_bg[fn] for fn in filelist]
                if options.labelfile_background else None)

    # print results
    if sys.stdout.isatty():
        BOLD = '\033[1m'
        UNBOLD = '\033[0m'
    else:
        BOLD = UNBOLD = ''
    print(("MAP: %s%.3f%s, " % (BOLD, results['map'], UNBOLD)) +
          (("MAP-bg: %.3f, " % results['map-bg'])
           if 'map-bg' in results else '') +
          ", ".join("top-%d: %.2f" % (i + 1, v)
                    for i, v in enumerate(results['top-acc'])))


if __name__ == "__main__":
    main()
