#!/bin/bash
# Author: Jan Schlüter
if [ $# -lt 1 ]; then
    echo "Calls the scripts of the subdirectories in correct order to prepare"
    echo "the BirdCLEF 2018 dataset for training and testing. It will need a"
    echo "lot of space for the downsampled audio files:"
    echo "- audio/train/2014: 12 GiB"
    echo "- audio/train/2015: 24 GiB"
    echo "- audio/train/2017: 18 GiB"
    echo "- audio/test: 17 GiB"
    echo "- audio/soundscapes: 1 GiB"
    echo "If you want them to be placed elsewhere, move the 'audio' directory"
    echo "somewhere and create a symlink here, or create symlinks 'audio/train',"
    echo "'audio/test' and 'audio/soundscapes' pointing to separate empty target"
    echo "directories. If you have multiple SDDs available, create separate"
    echo "'audio/train/2014', '/audio/train/2015' and 'audio/train/2017'"
    echo "symlinks to empty directories on different SDDs, to speed up training."
    echo "Usage: $0 SOURCE"
    echo "  SOURCE: The directory all BirdCLEF 2018 .tar.gz files have been"
    echo "      extracted to."
    exit 1
fi

here="${0%/*}"
source="$1"

echo "Converting audio files..."
"$here"/audio/recreate.sh "$source"
echo "Creating file lists..."
"$here"/filelists/recreate.sh
echo "Collecting xml files..."
"$here"/xml/recreate.sh "$source"
echo "Extracting labels and metadata..."
"$here"/labels/recreate.sh
