#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data feed definition for bird classification experiment.
Provides prepare_datafeed(), split_datafeed() and run_datafeed().

Author: Jan Schlüter
"""

import os
import io
try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

from progress import progress
import audio
import augment
from eval import softmax


def day_of_year(date):
    y, m, d = map(int, date.split('-', 2))
    return [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][m - 1] + d


def minute_of_day(time):
    h, m, s = map(int, time.split(':', 2))
    return 60 * h + m + s / 60.


def prepare_datafeed(filelist, datadir, designation, cfg):
    """
    Performs any preloading useful for run_datafeed(). Gives a list of file
    names to serve from, a base directory for the dataset, the designation
    ('train', 'valid', or 'test'), and a configuration dictionary. Returns some
    opaque object (to be passed to split_datafeed() or run_datafeed() via its
    first argument), and a dictionary mapping string names to dtypes and shapes
    of arrays that will be served by run_datafeed(). Shapes may contain `None`
    for non-fixed dimensions.
    """
    if designation not in ('train', 'valid', 'test'):
        raise ValueError('designation must be train, valid or test, not "%s"' %
                         designation)
    formats = {}

    # define generator for wav files
    sample_rate = cfg['sample_rate']
    wavs = (audio.WavFile(os.path.join(datadir, 'audio', fn),
                          sample_rate=sample_rate, channels=1, width=2)
            for fn in progress(filelist, 'File '))

    # for training and validation, pre-scan all wav files
    if designation in ('train', 'valid'):
        wavs = list(wavs)

    # we will later return spectrograms
    mel_max = cfg['mel_max']
    frame_len = cfg['frame_len']
    bin_nyquist = frame_len // 2 + 1
    if cfg['filterbank'] == 'mel_learn':
        bin_mel_max = bin_nyquist
    else:
        bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate
    formats['spect'] = (np.float32, (None, None, bin_mel_max))

    # and dictionaries of metadata, to be loaded in the following
    others = [{} for fn in filelist]

    # load labels if needed
    if designation in ('train', 'valid'):
        with io.open(os.path.join(datadir, 'labels', cfg['label_file'])) as f:
            labeldb = dict(l.rstrip('\r\n').split('\t') for l in f)
        labeldb = {fn: int(label) for fn, label in labeldb.items()}
        cfg['classes'] = max(labeldb.values()) + 1  # to save it with the model
        for fn, d in zip(filelist, others):
            d['label'] = labeldb[fn]
        formats['label'] = (np.intp, (None,))

    # load background labels if needed
    if designation in ('train', 'valid') and (cfg.get('cost_bg') or
                                              cfg.get('mix_fg_bg')):
        with io.open(os.path.join(datadir, 'labels', 'bg.tsv')) as f:
            labeldb_bg = dict(l.rstrip('\r\n').split('\t') for l in f)
        # we form 1-hot vectors of all background + the foreground species
        eye = np.eye(cfg['classes'], dtype=np.bool)
        labeldb_bg = {fn: np.any(np.stack([eye[int(l)]
                                           for l in label.split(',') if label]
                                          + [eye[labeldb[fn]]]),
                                 axis=0)
                      for fn, label in labeldb_bg.items()}
        for fn, d in zip(filelist, others):
            d['label_bg'] = labeldb_bg[fn]
        formats['label_bg'] = (np.bool, (None, cfg['classes']))

    # load additional metadata if needed
    if cfg.get('input_meta', ''):
        meta_required = cfg['input_meta'].split(',')
        with io.open(os.path.join(datadir, 'labels', 'meta.tsv')) as f:
            labeldb_meta = dict(l.rstrip('\r\n').split('\t', 1) for l in f)
        for fn, d in zip(filelist, others):
            m = {}
            (m['date'], m['time'], m['latitude'], m['longitude'],
             m['elevation']) = labeldb_meta[fn].split('\t')
            # delete all invalid or unneeded metadata items
            for k in list(m.keys()):
                if k not in meta_required or m[k] == '':
                    del m[k]
            # numerize all remaining metadata items
            if 'date' in m:
                m['date'] = float(day_of_year(m['date']))
            if 'time' in m:
                m['time'] = float(minute_of_day(m['time']))
            if 'latitude' in m:
                m['latitude'] = float(m['latitude'])
            if 'longitude' in m:
                m['longitude'] = float(m['longitude'])
            if 'elevation' in m:
                # in case the elevation is a range, we pick the upper bound
                m['elevation'] = float(m['elevation'].split('-', 1)[-1])
            d['meta'] = m
        meta_size = 0
        for k in 'latitude', 'longitude', 'elevation':
            if k in meta_required:
                meta_size += 2  # one for the value, one for the presence
        for k in 'date', 'time':
            if k in meta_required:
                meta_size += cfg['input_meta.%s_encoding.size' % k] + 1
        formats['meta'] = (np.float32, (None, meta_size))

    # append pseudo-labels if needed
    if designation == 'train' and cfg.get('pseudo_labels'):
        labelfiles, process = cfg['pseudo_labels'].split(':', 1)
        labelfiles = labelfiles.split(',')
        if process == 'softmax':
            process = softmax
        else:
            raise ValueError("unknown pseudo-label processing: %s" % process)
        labelfiles = [np.load(labelfile) for labelfile in labelfiles]
        for fn, d in zip(filelist, others):
            d['pseudo_label'] = process(sum(labelfile[fn].ravel()
                                            for labelfile in labelfiles) /
                                        float(len(labelfiles)))
        formats['pseudo_label'] = (np.float32, (None, cfg['classes']))

    # allow for mixed one-hot labels if needed
    if designation == 'train' and cfg['mixup2']:
        formats['mixed_label'] = (np.float32, (None, cfg['classes']))

    # load / compute class weights if needed
    if designation == 'train' and cfg.get('class_weights'):
        counts = np.bincount([d['label'] for d in others],
                             minlength=cfg['classes'])
        if cfg['class_weights'].startswith('google:'):
            # http://www.cs.tut.fi/sgn/arg/dcase2017/documents/workshop_presentations/the_story_of_audioset.pdf, p.30
            p_c = counts / float(len(filelist))
            p_bar = p_c.mean()
            beta = float(cfg['class_weights'].split(':', 1)[1])
            w = np.asarray((p_bar / p_c * (1 - p_c) / (1 - p_bar))**beta,
                           dtype=np.float32)
            for d in others:
                d['weight'] = w[d['label']]
            formats['weight'] = (np.float32, (None,))

    # sort by length if needed
    if designation != 'test':
        wavs, others = zip(*sorted(zip(wavs, others),
                                   key=lambda x: len(x[0])))

    return (wavs, others, designation), formats


def split_datafeed(datafeed, n, cfg):
    """
    Takes a prepared datafeed and splits it into `n` parts to be run in
    parallel. Returns a list of opaque objects accepted by run_datafeed().
    If unsupported, the list may just contain the single object given.
    """
    wavs, others, designation = datafeed
    if designation == 'test':
        # no useful split
        return [datafeed]
    elif designation == 'train':
        # can just run the same feed multiple times
        return [datafeed] * n
    elif designation == 'valid':
        # split into subsets that allow forming batches of uniform size, with
        # similar total sizes. wavs are already sorted by length, so we simply
        # assign contiguous batches to the n splits in a round-robin manner.
        batchsize = cfg['batchsize']
        data = list(izip(wavs, others))
        splits = [[] for _ in range(n)]
        num_batches = (len(data) + batchsize - 1) // batchsize
        for batch in range(num_batches):
            part = slice(batch * batchsize, (batch + 1) * batchsize)
            splits[batch % n].extend(data[part])
        return [tuple(zip(*wavs_others)) + (designation,)
                for wavs_others in splits if len(wavs_others) > 0]


def copy_metadata(batches):
    """
    Helper function yielding copies of the other and other['meta'] dicts so
    they can be modified in-place by later generators.
    """
    for spects, other in batches:
        other = list(other)
        for i in range(len(other)):
            other[i] = dict(other[i])
            other[i]['meta'] = dict(other[i]['meta'])
        yield spects, other


def encode_metadata(batches, cfg):
    """
    Helper function that encodes all metadata into a single float vector.
    """
    def encode_circular(value, maximum, size):
        # we map the value to the range [0, 1]
        value = float(value) / maximum
        # we choose size phases between 0 and pi (exclusive)
        phases = np.arange(size, dtype=np.float32) * np.pi / size
        # we produce an encoding from differently offset sinuoids,
        # scaled so the expected variance is 1.0
        return np.sqrt(2) * np.sin(2 * np.pi * value + phases)

    def encode_datum(datum, kind, cfg):
        if datum is None:
            if cfg['input_meta.missing'] == 'zero':
                if kind in ('date', 'time'):
                    return np.zeros(cfg['input_meta.%s_encoding.size' % kind],
                                    dtype=np.float32)
                else:
                    return np.zeros(1, dtype=np.float32)
            else:
                raise ValueError('unknown input_meta.missing="%s"' %
                                 cfg['input_meta.missing'])
        elif kind in ('date', 'time'):
            if cfg['input_meta.%s_encoding' % kind] == 'circular':
                return encode_circular(
                        datum,
                        maximum=365 if kind == 'date' else 24 * 60,
                        size=cfg['input_meta.%s_encoding.size' % kind])
            else:
                raise ValueError('unknown input_meta.%s_encoding = "%s"' %
                                 (kind, cfg['input_meta.%s_encoding' % kind]))
        else:
            shift = -cfg['input_meta.%s_mean' % kind]
            scale = 1. / cfg['input_meta.%s_std' % kind]
            datum = (datum + shift) * scale
            return np.array(datum, dtype=np.float32)[np.newaxis]

    meta_required = cfg['input_meta'].split(',')
    for spects, other in batches:
        for o in other:
            m = o['meta']
            present = np.array([k in m for k in meta_required],
                               dtype=np.float32)
            data = [encode_datum(m.get(k, None), k, cfg)
                    for k in meta_required]
            o['meta'] = np.concatenate([present] + data)
        yield spects, other


def pack_as_dict(batches, input_name='spect'):
    """
    Helper function converting mini-batches from (list of ndarray, list of dict)
    to dict of ndarray. `input_name` gives the key to use for the first ndarray.
    """
    def pack_array(array):
        if isinstance(array, (list, tuple)):
            if len(array) == 1 and isinstance(array[0], np.ndarray):
                return array[0][np.newaxis]
            return np.stack(array)
        return np.asarray(array)

    for spect, other in batches:
        batch = {input_name: pack_array(spect)}
        batch.update((k, pack_array([o[k] for o in other]))
                     for k in other[0].keys())
        yield batch


def run_datafeed(datafeed, cfg):
    """
    Yields batches of datapoints from a datafeed prepared by prepare_datafeed().
    Each batch takes the form of a dictionary mapping strings to numpy arrays of
    the dtype and shape indicated by prepare_datafeed(). A training datafeed
    will run indefinitely, a test or validation datafeed will run once.
    """
    wavs, others, designation = datafeed

    # grab some configuration information
    batchsize = cfg['batchsize']
    sample_rate = cfg['sample_rate']
    len_min = cfg['len_min']
    len_max = cfg['len_max']

    # create generators for batches of wavs
    if designation == 'train':
        len_buckets = cfg['len_buckets']
        factor = 2 if cfg['mixup'] else 1
        batches = augment.grab_random_excerpts(
                wavs, others, batchsize * factor, int(len_max * sample_rate),
                int(len_min * sample_rate), len_buckets)
        if cfg['mixup']:
            batches = augment.apply_mixup(batches, alpha=cfg['mixup'],
                                          dtype=np.float32)
        elif cfg['mixup2']:
            batches = augment.apply_mixup2(batches, classes=cfg['classes'],
                                           alpha=cfg['mixup2'],
                                           dtype=np.float32)
    elif designation == 'valid':
        def same_sized_batches(wavs, others):
            for b in range(0, len(wavs), batchsize):
                batch = wavs[b:b + batchsize]
                tgt_len = max(len(batch[0]), int(len(batch[-1]) * 0.8))
                tgt_len = min(max(tgt_len, len_min * sample_rate),
                              len_max * sample_rate)
                batch = [augment.loop(wav, tgt_len)
                         if len(wav) <= tgt_len
                         else wav[max(0, (len(wav) - tgt_len) // 2):
                                  (len(wav) + tgt_len) // 2]
                         for wav in batch]
                yield batch, others[b:b + batchsize]

        batches = same_sized_batches(wavs, others)

    elif designation == 'test':
        # single-item batches, looped to minimum length if needed
        batches = (([augment.loop(wav, len_min * sample_rate)], [other])
                   for wav, other in izip(wavs, others))

    # grab some more configuration information
    frame_len = cfg['frame_len']
    fps = cfg['fps']
    mel_max = cfg['mel_max']
    bin_nyquist = frame_len // 2 + 1
    if cfg['filterbank'] == 'mel_learn':
        bin_mel_max = bin_nyquist
    else:
        bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate

    # create generator for spectrogramming
    batches = augment.compute_spectrograms(
             batches, sample_rate, frame_len, fps, bins=bin_mel_max)

    # add data augmentation, if needed
    if designation == 'train' and cfg.get('spect_augment'):
        if cfg.get('max_db'):
            batches = augment.apply_random_filters(batches, mel_max,
                                                   cfg['max_db'])

    # drop/augment/encode metadata, if needed
    if cfg['input_meta']:
        batches = copy_metadata(batches)  # to allow in-place modifications
        if designation == 'train':
            if cfg['input_meta.dropout']:
                batches = augment.drop_metadata(
                        batches, cfg['input_meta.dropout'])
            for k in 'date', 'time', 'latitude', 'longitude', 'elevation':
                if k in cfg['input_meta'] and cfg['input_meta.%s_noise' % k]:
                    batches = augment.blur_metadata(
                            batches, k, cfg['input_meta.%s_noise' % k])
        batches = encode_metadata(batches, cfg)

    # move from (spect, other) tuples to dictionaries
    batches = pack_as_dict(batches)
    return batches
