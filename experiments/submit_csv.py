#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converts a prediction file into the CSV submission format.

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
    descr = "Converts a prediction file into the CSV submission format."
    parser = ArgumentParser(description=descr)
    parser.add_argument('infile', metavar='INFILE',
            type=str,
            help='File to load the prediction curves from (.npz/.pkl format).')
    parser.add_argument('outfile', metavar='OUTFILE',
            type=str,
            help='File to save the predictions to (.csv format).')
    parser.add_argument('--dataset',
            type=str, default='birdclef',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--filelist',
            type=str, default='test',
            help='Name of the file list to use (default: %(default)s)')
    parser.add_argument('--softmax',
            action='store_true', default=True,
            help='If given, apply a softmax to the predictions (default).')
    parser.add_argument('--no-softmax',
            action='store_false', dest='softmax',
            help='If given, do not apply a softmax to the predictions.')
    parser.add_argument('--sigmoid',
            action='store_true', default=True,
            help='If given, apply a sigmoid to the predictions.')
    parser.add_argument('--mode',
            type=str, choices=('mono', 'soundscape'), default='mono',
            help='Whether to prepare for the monodirectional or soundscape '
                 'task. (default: %(default)s)')
    parser.add_argument('--max-classes',
            type=int, default=100,
            help='Maximum number of classes to output per item (default: '
                 '%(default)s)')
    return parser


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


def hour_min_sec(sec):
    return sec // 3600, (sec % 3600) // 60, sec % 60


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    infile = options.infile
    outfile = options.outfile

    # load file lists
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)
    with io.open(os.path.join(datadir, 'filelists', options.filelist)) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]

    # load label names
    with io.open(os.path.join(datadir, 'labels', 'labelset')) as f:
        labelset = [l.rstrip() for l in f if l.rstrip()]

    # load network predictions
    preds = np.load(infile)
    if options.mode == 'mono':
        preds = {fn: preds[fn].ravel() for fn in filelist}
    if options.softmax:
        preds = {fn: softmax(preds[fn]) for fn in filelist}
    if options.sigmoid:
        preds = {fn: sigmoid(preds[fn]) for fn in filelist}
    if next(iter(preds.values())).shape[-1] != len(labelset):
        raise ValueError("Expected %d classes, got %d instead" %
                         (len(labelset), next(iter(preds.values())).shape[-1]))

    # write CSV file
    with io.open(outfile, 'wb' if sys.version_info[0] == 2 else 'w') as f:
        for fn in filelist:
            media_id = int(fn.rsplit('.', 1)[0].rsplit('_RN', 1)[1])
            if options.mode == 'mono':
                f.writelines('%d;%s;%.10f;%d\n' %
                             (media_id, labelset[species], preds[fn][species], rank + 1)
                             for rank, species in enumerate(
                                     np.argsort(preds[fn])[::-1][:options.max_classes]))
            elif options.mode == 'soundscape':
                f.writelines('%d;%02d:%02d:%02d-%02d:%02d:%02d;%s;%.10f\n' %
                             ((media_id,) + hour_min_sec(5 * i) + hour_min_sec(5 * i + 5) + (labelset[species], p[species]))
                             for i, p in enumerate(preds[fn])
                             for species in np.argsort(p)[::-1][:options.max_classes]
                             if p[species] > 0)


if __name__ == "__main__":
    main()
