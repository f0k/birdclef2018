#!/bin/bash
# Author: Jan Schlüter
if [ $# -lt 1 ]; then
    echo "Calls collect.sh multiple times for a common source directory all"
    echo "the BirdCLEF .tar.gz files have been extracted to."
    echo "Usage: $0 SOURCE"
    echo "  SOURCE: The directory all BirdCLEF 2018 .tar.gz files have been"
    echo "      extracted to."
    exit 1
fi

here="${0%/*}"
source="$1"

# The downloaded files follow different naming schemes, but luckily the
# root directories in each are still uniquely named.

# BirdCLEF2017TrainingSetPart1.tar.gz
# BirdCLEF2017TrainingSetPart2.tar.gz
"$here"/collect.sh --split-years "$here"/train \
        "$source"/TrainingSet "$source"/data

# BirdCLEF2018MonophoneTest.tar.gz
"$here"/collect.sh --split-years "$here"/test \
        "$source"/BirdCLEF2018MonophoneTest

# BirdCLEF2018SoundscapesValidation.tar.gz
"$here"/collect.sh "$here"/soundscapes/val \
        "$source"/val/data

# BirdCLEF2018SoundscapesTest.tar.gz
"$here"/collect.sh "$here"/soundscapes/test \
        "$source"/test
