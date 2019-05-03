#!/bin/bash
# Author: Jan Schlüter
if [ $# -lt 2 ]; then
    echo "Collects BirdCLEF wav files into a target directory from one or"
    echo "multiple source directories, converting to 22 kHz mono WAV files on"
    echo "the way. Runs 8 conversion processes in parallel. Needs ffmpeg."
    echo "Usage: $0 [--split-years] TARGET SOURCE..."
    echo "  --split-years: If given, will create a subdirectory per year to"
    echo "     keep the directory sizes low. Must be the first argument."
    echo "  TARGET: The target directory; will be created if needed, meant to"
    echo "     be one of train, test, soundscapes/test, soundscapes/val"
    echo "  SOURCE: The directory to browse recursively for .wav files, can"
    echo "     be given multiple times"
fi

if [ "$1" == "--split-years" ]; then
    split_years=1
    shift 1
else
    split_years=0
fi

target="$1"
i=1
while IFS= read -d '' -r infile; do
    base="${infile##*/}"
    if [ "$split_years" -eq 1 ]; then
        year="${base:8:4}"
        outdir="$target/$year"
    else
        outdir="$target"
    fi
    outfile="$outdir/$base"
	if [ ! -f "$outfile" ]; then
    	mkdir -p "$outdir"
		# display progress on stderr
		>&2 echo -ne "\r\e[K$i: $outfile"  # \r: return, \e[K: delete rest of line
		# write command to stdout (0-terminated)
		echo -ne "ffmpeg -v quiet -nostdin -i \"$infile\" -c:a pcm_s16le -ac 1 -ar 22050 \"$outfile\"\0"
	fi
	((i++))
done < <(find "${@:2}" -name '*.wav' -print0) | xargs --no-run-if-empty -0 -n1 -P8 sh -c  # execute up to eight commands in parallel
>&2 echo
