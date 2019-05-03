#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio feature extraction routines.

Author: Jan Schlüter
"""

import subprocess
import wave

import numpy as np
try:
    from pyfftw.builders import rfft as rfft_builder
    from pyfftw import empty_aligned
except ImportError:
    def rfft_builder(samples, *args, **kwargs):
        if samples.dtype == np.float32:
            return lambda *a, **kw: np.fft.rfft(*a, **kw).astype(np.complex64)
        else:
            return np.fft.rfft
    empty_aligned = np.empty


def read_ffmpeg(infile, sample_rate, cmd='ffmpeg'):
    """
    Decodes a given audio file using ffmpeg, resampled to a given sample rate,
    downmixed to mono, and converted to float32 samples. Returns a numpy array.
    """
    call = [cmd, "-v", "quiet", "-i", infile, "-f", "f32le",
            "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
    samples = subprocess.check_output(call)
    return np.frombuffer(samples, dtype=np.float32)


def spectrogram(samples, sample_rate, frame_len, fps, batch=48, dtype=None,
                bins=None, plans=None):
    """
    Computes a magnitude spectrogram for a given vector of samples at a given
    sample rate (in Hz), frame length (in samples) and frame rate (in Hz).
    Allows to transform multiple frames at once for improved performance (with
    a default value of 48, more is not always better). Returns a numpy array.
    Allows to return a limited number of bins only, with improved performance
    over discarding them afterwards. Optionally accepts a set of precomputed
    plans created with spectrogram_plans(), required when multi-threading.
    """
    if dtype is None:
        dtype = samples.dtype
    if bins is None:
        bins = frame_len // 2 + 1
    if len(samples) < frame_len:
        return np.empty((0, bins), dtype=dtype)
    if plans is None:
        plans = spectrogram_plans(frame_len, batch, dtype)
    rfft1, rfft, win = plans
    hopsize = int(sample_rate // fps)
    num_frames = (len(samples) - frame_len) // hopsize + 1
    nabs = np.abs
    naa = np.asanyarray
    if batch > 1 and num_frames >= batch and samples.flags.c_contiguous:
        frames = np.lib.stride_tricks.as_strided(
                samples, shape=(num_frames, frame_len),
                strides=(samples.strides[0] * hopsize, samples.strides[0]))
        spect = [nabs(rfft(naa(frames[pos:pos + batch:], dtype) * win)[:, :bins])
                 for pos in range(0, num_frames - batch + 1, batch)]
        samples = samples[(num_frames // batch * batch) * hopsize::]
        num_frames = num_frames % batch
    else:
        spect = []
    if num_frames:
        spect.append(np.vstack(
                [nabs(rfft1(naa(samples[pos:pos + frame_len:],
                                dtype) * win)[:bins:])
                 for pos in range(0, len(samples) - frame_len + 1, hopsize)]))
    return np.vstack(spect) if len(spect) > 1 else spect[0]


def spectrogram_plans(frame_len, batch=48, dtype=np.float32):
    """
    Precompute plans for spectrogram(), for a given frame length, batch size
    and dtype. Returns two plans (single spectrum and batch), and a window.
    """
    input_array = empty_aligned((batch, frame_len), dtype=dtype)
    win = np.hanning(frame_len).astype(dtype)
    return (rfft_builder(input_array[0]), rfft_builder(input_array), win)


def extract_spect(filename, sample_rate=22050, frame_len=1024, fps=70):
    """
    Extracts a magnitude spectrogram for a given audio file at a given sample
    rate (in Hz), frame length (in samples) and frame rate (in Hz). Returns a
    numpy array.
    """
    try:
        samples = read_ffmpeg(filename, sample_rate)
    except Exception:
        samples = read_ffmpeg(filename, sample_rate, cmd='avconv')
    return spectrogram(samples, sample_rate, frame_len, fps)


def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq,
                          max_freq):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples.
    """
    # prepare output matrix
    input_bins = (frame_len // 2) + 1
    filterbank = np.zeros((input_bins, num_bands))

    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    spacing = (max_mel - min_mel) / (num_bands + 1)
    peaks_mel = min_mel + np.arange(num_bands + 2) * spacing
    peaks_hz = 700 * (np.exp(peaks_mel / 1127) - 1)
    fft_freqs = np.linspace(0, sample_rate / 2., input_bins)
    peaks_bin = np.searchsorted(fft_freqs, peaks_hz)

    # fill output matrix with triangular filters
    for b, filt in enumerate(filterbank.T):
        # The triangle starts at the previous filter's peak (peaks_freq[b]),
        # has its maximum at peaks_freq[b+1] and ends at peaks_freq[b+2].
        left_hz, top_hz, right_hz = peaks_hz[b:b + 3]  # b, b+1, b+2
        left_bin, top_bin, right_bin = peaks_bin[b:b + 3]
        # Create triangular filter compatible to yaafe
        filt[left_bin:top_bin] = ((fft_freqs[left_bin:top_bin] - left_hz) /
                                  (top_bin - left_bin))
        filt[top_bin:right_bin] = ((right_hz - fft_freqs[top_bin:right_bin]) /
                                   (right_bin - top_bin))
        filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()

    return filterbank


class WavFile(object):
    """
    Encapsulates a RIFF wave file providing memmapped access to its samples.
    If `sample_rate`, `channels` or `width` are given, a RuntimeError is raised
    if it does not match the file's format.
    """
    def __init__(self, filename, sample_rate=None, channels=None, width=None):
        self.filename = filename
        with open(filename, 'rb') as f:
            try:
                w = wave.open(f, 'r')
            except Exception as e:
                raise RuntimeError("could not read %s: %r" % (filename, e))
            self.sample_rate = w.getframerate()
            self.channels = w.getnchannels()
            self.width = w.getsampwidth() // self.channels
            self.length = w.getnframes()
            self.offset = w._data_chunk.offset + 8  # start of samples in file
        if (sample_rate is not None) and (sample_rate != self.sample_rate):
            raise RuntimeError("%s has sample rate %d Hz, wanted %d" %
                               (self.sample_rate, sample_rate))
        if (channels is not None) and (channels != self.channels):
            raise RuntimeError("%s has %d channel(s), wanted %d" %
                               (self.channels, channels))
        if (width is not None) and (width != self.width):
            raise RuntimeError("%s has sample width %d byte(s), wanted %d" %
                               (self.width, width))

    @property
    def shape(self):
        return (self.length, self.channels)

    @property
    def dtype(self):
        return {1: np.int8, 2: np.int16}[self.width]

    @property
    def samples(self):
        """
        Read-only access of the samples as a memory-mapped numpy array.
        """
        return np.memmap(self.filename, self.dtype, mode='r',
                         offset=self.offset, shape=self.shape)

    def __len__(self):
        return self.length

    def __array__(self, *args):
        return np.asanyarray(self.samples, *args)

    def __getitem__(self, obj):
        return self.samples[obj]
