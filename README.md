Bird Identification from Timestamped, Geotagged Audio Recordings
================================================================

This is the implementation of the
[BirdCLEF 2018](http://www.imageclef.org/node/230) submission by
[OFAI](http://www.ofai.at) within the [aMOBY](http://amoby.ofai.at) project.

It allows training an ensemble of neural networks to recognize 1500 South
American bird species in audio recordings, with an option to factor in
metadata about the recording date, time and location.

It contains the code for preparing the dataset (converting audio files and
parsing metadata), for training a set of different models on audio recordings
and/or metadata, for finding weights to form an ensemble of those models, and
for producing the predictions on the test set for submission to the challenge.

For a detailed description of the approach, please refer to the paper
"Bird Identification from Timestamped, Geotagged Audio Recordings"
by Jan Schlüter included in the CLEF Working Notes 2018.
[[Paper](http://ofai.at/~jan.schlueter/pubs/2018_birdclef.pdf),
 [BibTeX](http://ofai.at/~jan.schlueter/pubs/2018_birdclef.bib)]


Preliminaries
-------------

The code requires the following software:
* Python 2.7+ or 3.4+
* Python packages: numpy, scipy, Theano, Lasagne
* bash or a compatible shell
* ffmpeg

For better performance, the following Python packages are recommended:
* pyfftw (for much faster spectrogram computation)

Before installing the dependencies, if desired, create and activate an
environment using `pyenv` and/or `virtualenv`/`venv`, or using `conda`.

Install the bleeding-edge versions of Theano and Lasagne from github:
```bash
pip install --upgrade --no-deps https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade --no-deps https://github.com/Lasagne/Lasagne/archive/master.zip
```
(If not in an environment, add `--user` to install in your home directory, or
`sudo` to install globally.)

For GPU support, also install libgpuarray, following its [installation
instructions](http://deeplearning.net/software/libgpuarray/installation.html).
For a more complete guide including CUDA and cuDNN, please refer to the [From
Zero to Lasagne](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne)
guides.

For faster FFTs, install libfftw3 and pyfftw. On Ubuntu, this can be done with:
```bash
sudo apt-get install libfftw3-dev
pip install pyfftw
```
Under conda, it would be:
```bash
conda install -c conda-forge pyfftw
```


Setup
-----

For preparing the experiments, clone the repository somewhere:
```bash
git clone https://github.com/f0k/birdclef2018.git
```
If you do not have `git` available, download the code from
https://github.com/f0k/birdclef2018/archive/master.zip and extract it.

The experiments rely on the BirdCLEF 2018 dataset. First download the files
(specifically, BirdCLEF2017TrainingSetPart1.tar.gz,
BirdCLEF2017TrainingSetPart2.tar.gz, BirdCLEF2018MonophoneTest.tar.gz,
BirdCLEF2018SoundscapesTest.tar.gz, BirdCLEF2018SoundscapesValidation.tar.gz)
and extract them to a common directory. If you were not a BirdCLEF participant,
ask the organizers if they are willing to share the URLs.

Then open the cloned or extracted repository in a bash terminal and execute the
following:
```bash
./datasets/birdclef/recreate.sh
```
It will tell you that you need to specify the path to the extracted files, but
it will also display some useful hints on how to organize the placement of the
converted audio files.
This script will call other scripts to convert the audio to 22 kHz mono files
(this saves time during training), build the file lists for training and
testing, and extract the ground truth and metadata from the XML files.

Finally, for all following commands, go into the `experiments` directory:
```bash
cd experiments
```


Training
--------

To train all models for the ensemble, simply run:
```bash
./train_all.sh
```
To use a GPU, either setup a `.theanorc` file in your home directory, or run:
```bash
THEANO_FLAGS=device=cuda,floatX=float32,gpuarray.preallocate=11000 ./train_all.sh
```
This will train 17 audio, 19 metadata and one combined network(s). On an Nvidia
Titan X Pascal GPU, a single training run will take up to 10 hours for audio,
and 50 minutes for metadata networks. If your GPU does not have enough memory,
reduce or remove the `gpuarray.preallocate=11000` setting, and reduce the batch
size that is set in the `defaults.vars` file.

If you have multiple GPUs, you can distribute runs over these GPUs by running
the script multiple times in multiple terminals with different target devices,
e.g., `THEANO_FLAGS=device=cuda1 ./train_all.sh`. If you have multiple servers
that can access the same directory via NFS, you can also run the script on
each server for further distribution of runs (runs are blocked with lockfiles).

The script will also compute network predictions after each training run. If
this failed for some jobs for some reasons, run:
```bash
./predict_missing.sh
```
This will compute any missing network predictions (if none are missing, nothing
happens).


Evaluation
----------

To obtain results for all networks trained so far, run:
```bash
./eval_all.sh
```
This will print the Mean Average Precision (MAP) against the foreground species,
the MAP against the background species, and the top-*k* accuracy for the
foreground species for *k* between 1 and 5, all on the validation set (the test
set is kept secret by the organizers of the BirdCLEF challenge).


Ensembling
----------

After all models have been trained, you can run `hyperopt` to find an optimal
linear combination of models based on the validation set performance. Install
it with:
```bash
pip install hyperopt
```
We can now run `blender.py` to do the actual optimization. The commands are
documented in comments in `submit_all.sh`. For example, for the audio-only
ensemble, run:
```bash
./blender.py --dataset=birdclef --labelfile-background=bg.tsv --strategy=hyperopt \
  birdclef/{dummy,resnet1}_{lme1,att16,att64}_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1{,_mixfgbg}{,_ban1}.pred.pkl
```
In the end, it will produce a list of selected models and combination weights
that can be directly copied to `submit_all.sh`, preceded by `submit` and a name
for the ensemble. It can also be used directly as arguments to `./eval.py` to
evaluate the ensemble.


Submission
----------

Finally, to create the CSV files for submission, run:
```bash
./submit_all.sh
```
Prefix the command with a `THEANO_FLAGS=...` setting if needed.
This will compute predictions on the test set for all models participating in
any of the ensembles, combine the predictions according to the weights, and
produce a CSV file for each ensemble.


Reusing
-------

### ... for different datasets

Datasets can be added to the `datasets` directory and their name be passed as
the `--dataset` argument of `train.py`, `predict.py`, `eval.py` (and
`blender.py`, if needed). Each dataset directory must contain:
* an `audio` subdirectory with `.wav` files (this is a strict requirement, since
  they are accessed as memory maps),
* a `filelists` directory with at least a `train` and `valid` file listing the
  file names relative to the `audio` directory, and
* a `labels` directory with a `fg.tsv` file listing the training and validation
  file names along with their class labels, with a tab character in between, and
  a `labelset` file listing all class names to give them a fixed order.

### ... for different frameworks

The implementation makes some use of features unique to Lasagne, so it is not
trivial to port completely to another framework. Some parts may be interesting
to take out, though:
* `audio.py` contains code for fast spectrogram computation, and a `WavFile`
  class for masquerading a `.wav` files as a numpy array that is lazily mapped
  to memory when needed.
* `augment.py` contains `grab_random_excerpts()`, which provides a way to yield
  random excerpts from a set of audio files with wildly different lengths. Each
  mini-batch will have same-length excerpts, with the length bounded between a
  given minimum and maximum length, and files drawn from buckets to avoid
  excessive cropping or padding.
* `model.py` contains a learnable mel filterbank, a learnable magnitude
  transformation, PCEN, and log-mean-exp pooling
* `model_to_fcn.py` implements a conversion of a CNN that classifies excerpts to
  a fully-convolutional network with dilated convolutions and dilated max-pooling
  that efficiently processes a full recording, keeping the full output resolution
