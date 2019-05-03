#!/bin/bash
here="${0%/*}"
pushd "$here/../audio"
# The validation set was produced by Stefan Kahl in 2016. We're reusing it here.
# We collect all training files except the ones that are in the validation set:
find -L train/ -name '*.wav' | grep -vf ../filelists/valid | sort -g > /tmp/train
# And all test files:
find -L test/ -name '*.wav' | sort -g > ../filelists/test
# And the same for the soundscapes:
find -L soundscapes/val/ -name '*.wav' | sort -g > ../filelists/soundscapes_val
# And the same for the soundscapes:
find -L soundscapes/test/ -name '*.wav' | sort -g > ../filelists/soundscapes_test
popd
