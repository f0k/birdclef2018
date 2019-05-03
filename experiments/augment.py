#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data augmentation routines.

Author: Jan Schlüter
"""

import numpy as np
import scipy.ndimage

import audio


def loop(array, length):
    """
    Loops a given `array` along its first axis to reach a length of `length`.
    """
    if len(array) < length:
        array = np.asanyarray(array)
        if len(array) == 0:
            return np.zeros((length,) + array.shape[1:], dtype=array.dtype)
        factor = length // len(array)
        if factor > 1:
            array = np.tile(array, (factor,) + (1,) * (array.ndim - 1))
        missing = length - len(array)
        if missing:
            array = np.concatenate((array, array[:missing:]))
    return array


def crop(array, length):
    """
    Crops a random excerpt of `length` along the first axis of `array`.
    """
    if len(array) > length:
        pos = np.random.randint(len(array) - length + 1)
        array = array[pos:pos + length:]
    return array


def grab_random_excerpts(wavs, labels, batchsize, max_length, min_length=0,
                         buckets=3):
    """
    Extracts random excerpts from the signals in `wavs` paired with the label
    from `labels` associated with the file. Yields `batchsize` excerpts at a
    time. All excerpts in a batch have the same size, which can vary between
    `min_length` and `max_length` (in samples). Shorter signals are looped,
    longer signals are cropped. To avoid excessive looping or cropping, all
    excerpts in a batch are drawn from one of `buckets` many groups of similar
    length. Each file has the same probability of being selected.
    """
    # sort by length
    wavs, labels = zip(*sorted(zip(wavs, labels), key=lambda x: len(x[0])))
    # cut into buckets
    lens = np.fromiter(map(len, wavs), dtype=np.int, count=len(wavs))
    low, up = np.searchsorted(lens, (min_length, max_length))
    if low > 0:
        buckets -= 1  # reserve one bucket for smaller than minimum
    if up < len(wavs):
        buckets -= 1  # reserve one bucket for larger than maximum
    bounds = low + np.arange(buckets + 1) * (up - low) / float(buckets)
    bounds = np.concatenate(([0], bounds.astype(np.int), [len(wavs)]))
    sizes = np.diff(bounds)
    proportions = sizes / float(len(wavs))
    # infinite loop
    while True:
        # pick bucket
        bucket = np.random.choice(len(proportions), p=proportions)
        # pick files within bucket (without replacement)
        idxs = (np.random.choice(sizes[bucket], batchsize, replace=False) +
                bounds[bucket])
        # collect
        batch_wavs = [wavs[idx] for idx in idxs]
        batch_labels = [labels[idx] for idx in idxs]
        # loop or crop
        if max_length == 0:  # special case for training on metadata only
            batch_wavs = [np.empty((0,) + wav.shape[1:], dtype=wav.dtype)
                          for wav in batch_wavs]
        elif bucket == 0 and low > 0:
            batch_wavs = [loop(wav, min_length) for wav in batch_wavs]
        else:
            tgt_length = min(min(len(wav) for wav in batch_wavs), max_length)
            batch_wavs = [crop(wav, tgt_length) for wav in batch_wavs]
            # explicitly copy memory-mapped views to memory to close files
            # batch_wavs = [np.array(wav) for wav in batch_wavs]
        yield batch_wavs, batch_labels


def apply_mixup(batches, alpha=0.2, dtype=None):
    """
    Applies mixup augmentation, mixing the second half of the batch with the
    first and discarding it, with mixing weights from Beta(alpha, alpha+1).
    Paper: https://arxiv.org/abs/1710.09412; here we use the reformulation in
    http://www.inference.vc/mixup-data-dependent-data-augmentation/.
    """
    one = np.ones((), dtype=dtype)
    for wavs, labels in batches:
        batchsize = len(wavs) // 2
        weights = np.random.beta(alpha + 1, alpha, batchsize).astype(dtype)
        if isinstance(wavs, np.ndarray):
            weights = weights[(slice(None),) + (np.newaxis,) * (wavs.ndim - 1)]
            wavs = np.asanyarray(wavs, dtype=dtype)
            wavs = (wavs[:batchsize] * weights +
                    wavs[batchsize:] * (one - weights))
        else:
            wavs = [np.multiply(wav1, weight, dtype=dtype) +
                    np.multiply(wav2, one - weight, dtype=dtype)
                    for wav1, wav2, weight
                    in zip(wavs[:batchsize], wavs[batchsize:], weights)]
        yield wavs, labels[:batchsize]


def apply_mixup2(batches, classes, alpha=0.2, dtype=None):
    """
    Applies mixup augmentation, mixing the second half of the batch with the
    first and discarding it, both for the inputs and for the labels (which are
    converted to 1-hot encoding), with mixing weights from Beta(alpha, alpha).
    Paper: https://arxiv.org/abs/1710.09412; here we use their formulation.
    """
    eye = np.eye(classes, dtype=dtype)
    one = np.ones((), dtype=dtype)
    for wavs, labels in batches:
        batchsize = len(wavs) // 2
        labels = eye[labels]  # to one-hot encoding
        weights = np.random.beta(alpha, alpha, batchsize).astype(dtype)
        if isinstance(wavs, np.ndarray):
            weights = weights[(slice(None),) + (np.newaxis,) * (wavs.ndim - 1)]
            wavs = np.asanyarray(wavs, dtype=dtype)
            wavs = (wavs[:batchsize] * weights +
                    wavs[batchsize:] * (one - weights))
            weights = weights.ravel()[:, np.newaxis]
            labels = (labels[:batchsize] * weights +
                      labels[batchsize:] * (one - weights))
        else:
            wavs = [np.multiply(wav1, weight, dtype=dtype) +
                    np.multiply(wav2, one - weight, dtype=dtype)
                    for wav1, wav2, weight
                    in zip(wavs[:batchsize], wavs[batchsize:], weights)]
            labels = [np.multiply(label1, weight, dtype=dtype) +
                      np.multiply(label2, one - weight, dtype=dtype)
                      for label1, label2, weight
                      in zip(labels[:batchsize], labels[batchsize:], weights)]
        yield wavs, labels


def compute_spectrograms(batches, sample_rate, frame_len, fps, bins=None):
    """
    Computes spectrograms from the signals in `batches` at a given sample rate
    (in Hz), frame length (in samples) and frame rate (in Hz).
    """
    plans = audio.spectrogram_plans(frame_len, dtype=np.float32)
    for wavs, labels in batches:
        spects = [audio.spectrogram(np.asanyarray(wav).ravel(),
                                    sample_rate, frame_len, fps,
                                    dtype=np.float32, bins=bins, plans=plans)
                  for wav in wavs]
        yield spects, labels


def apply_random_stretch_shift(batches, max_stretch, max_shift,
                               keep_frames, keep_bins, order=3,
                               prefiltered=False):
    """
    Apply random time stretching of up to +/- `max_stretch`, random pitch
    shifting of up to +/- `max_shift`, keeping the central `keep_frames` frames
    and the first `keep_bins` bins. For performance, the spline `order` can be
    reduced, and inputs can be `prefiltered` with scipy.ndimage.spline_filter.
    """
    for spects, labels in batches:
        outputs = np.empty((len(spects), keep_frames, keep_bins),
                           dtype=spects.dtype)
        randoms = (np.random.rand(len(spects), 2) - .5) * 2
        for spect, output, random in zip(spects, outputs, randoms):
            stretch = 1 + random[0] * max_stretch
            shift = 1 + random[1] * max_shift
            # We can do shifting/stretching and cropping in a single affine
            # transform (including setting the upper bands to zero if we shift
            # down the signal so far that we exceed the input's nyquist rate)
            scipy.ndimage.affine_transform(
                    spect, (1 / stretch, 1 / shift),
                    output_shape=(keep_frames, keep_bins), output=output,
                    offset=(.5 * (len(spect) * stretch - keep_frames), 0),
                    mode='constant', cval=0, order=order,
                    prefilter=not prefiltered)
        # clip possible negative values introduced by the interpolation
        np.maximum(outputs, 0, outputs)
        yield outputs, labels


def apply_filterbank(batches, filterbank):
    """
    Apply a filterbank to batches of spectrogram excerpts via a dot product.
    """
    for spects, labels in batches:
        # we reshape (batchsize, frames, bins) to (batchsize * frames, bins) so
        # we can transform all excerpts in a single dot product, then go back
        # to (batchsize, frames, filters)
        spects = np.asanyarray(spects)
        yield (np.dot(spects.reshape(-1, spects.shape[-1]), filterbank).reshape(
                (spects.shape[0], spects.shape[1], -1)), labels)


def apply_logarithm(batches):
    """
    Convert linear to logarithmic magnitudes, as log(1 + x).
    """
    for spects, labels in batches:
        yield np.log1p(np.maximum(spects, 0)), labels


def apply_random_filters(batches, max_freq, max_db, min_std=5, max_std=7):
    """
    Applies random filter responses to logarithmic-magnitude mel spectrograms.
    The filter response is a Gaussian of a standard deviation between `min_std`
    and `max_std` semitones, a mean between 150 Hz and `max_freq`, and a
    strength between -/+ `max_db` dezibel. Assumes the spectrograms cover up to
    `max_freq` Hz.
    """
    for spects, labels in batches:
        if not isinstance(spects, np.ndarray):
            spects = np.stack(spects)
        batchsize, length, bins = spects.shape
        # sample means and std deviations on logarithmic pitch scale
        min_pitch = 12 * np.log2(150)
        max_pitch = 12 * np.log2(max_freq)
        mean = min_pitch + (np.random.rand(batchsize) *
                            (max_pitch - min_pitch))
        std = min_std + np.random.randn(batchsize) * (max_std - min_std)
        # convert means and std deviations to linear frequency scale
        std = 2**((mean + std) / 12) - 2**(mean / 12)
        mean = 2**(mean / 12)
        # convert means and std deviations to bins
        mean = mean * bins / max_freq
        std = std * bins / max_freq
        # sample strengths uniformly in dB
        strength = max_db * 2 * (np.random.rand(batchsize) - .5)
        # create Gaussians
        filt = (strength[:, np.newaxis] *
                np.exp(np.square((np.arange(bins) - mean[:, np.newaxis]) /
                                 std[:, np.newaxis]) * -.5))
        # transform from dB to factors
        filt = 10**(filt / 20.)
        # apply (multiply, broadcasting over the second axis)
        filt = np.asarray(filt, dtype=spects.dtype)
        yield spects * filt[:, np.newaxis, :], labels


def apply_znorm(batches, mean, istd):
    """
    Apply Z-scoring (subtract mean, multiply by inverse std deviation).
    """
    for spects, labels in batches:
        yield (spects - mean) * istd, labels


def drop_metadata(batches, p):
    """
    Drop metadata items with a probability of `p`. Modifies in-place.
    """
    for spects, other in batches:
        drop = (np.random.rand(sum(len(o['meta']) for o in other)) < p)
        drop = iter(drop)
        for o in other:
            for k in list(o['meta'].keys()):
                if next(drop):
                    del o['meta'][k]
        yield spects, other


def blur_metadata(batches, name, stddev):
    """
    Add Gaussian noise to a given metadata item with a given standard deviation.
    Modifies in-place.
    """
    for spects, other in batches:
        for o in other:
            if name in o['meta']:
                o['meta'][name] += np.random.randn() * stddev
        yield spects, other


def generate_in_background(generators, num_cached=50, in_processes=False):
    """
    Runs generators in background threads or processes, caching up to
    `num_cached` items. Multiple generators are each run in a separate
    thread/process, and their items will be interleaved in unpredictable order.
    """
    if not in_processes:
        try:
            from Queue import Queue
        except ImportError:
            from queue import Queue
        from threading import Thread as Background
        sentinel = object()  # guaranteed unique reference
    else:
        from multiprocessing import Queue
        from multiprocessing import Process as Background
        sentinel = None  # object() would be different between processes

    queue = Queue(maxsize=num_cached)

    # define producer (putting items into queue)
    def producer(generator, queue, sentinel, seed=None):
        if seed is not None:
            np.random.seed(seed)
        try:
            for item in generator:
                queue.put(item)
            queue.put(sentinel)
        except Exception as e:
            import traceback
            queue.put(type(e)('%s caught in background.\nOriginal %s' %
                              (e.args[0], traceback.format_exc())))

    # start producers (in background threads or processes)
    active = 0
    for generator in generators:
        # when multiprocessing, ensure processes have different random seeds
        seed = np.random.randint(2**32 - 1) if in_processes and active else None
        bg = Background(target=producer,
                        args=(generator, queue, sentinel, seed))
        bg.daemon = True
        bg.start()
        active += 1
        if active > num_cached:
            raise ValueError("generate_in_background() got more generators "
                             "than cached items (%d). Make sure you supplied "
                             "a list or iterable of generators as the first "
                             "argument, not a generator." % num_cached)

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while True:
        if item is sentinel:
            active -= 1
            if not active:
                break
        elif isinstance(item, Exception):
            raise item
        else:
            yield item
        item = queue.get()
