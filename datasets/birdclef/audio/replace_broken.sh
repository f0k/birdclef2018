#!/bin/bash
# Author: Jan Schlüter
if [ $# -lt 1 ]; then
    echo "Replaces zero-length .wav files with 0.1 seconds of silence."
    echo "Usage: $0 DIR [...]"
    echo "  DIR: The directory to process."
    exit 1
fi

# Use find and ffmpeg to replace all 0-length files with 0.1s of silence.
find -L "$@" -size 0 -name '*.wav' -delete -exec ffmpeg -v quiet -nostdin -f lavfi -i anullsrc=r=22050:cl=mono -t 0.1 -c:a pcm_s16le {} \;
