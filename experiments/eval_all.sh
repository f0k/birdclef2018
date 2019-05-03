#!/bin/bash

# Evaluates all models; or all models given on the command line.

here="${0%/*}"
if [ "$1" != "" ]; then
	for predfile in "$@"; do
		if [ "${predfile##*.}" == 'vars' ]; then
			predfile="${predfile%npz.vars}pred.pkl"
		elif [ "${predfile##*.}" == 'npz' ]; then
			predfile="${predfile%npz}pred.pkl"
		fi
		if [ "${predfile%.pred.pkl}" != "$predfile" ]; then
			echo "$predfile"
			"$here"/eval.py "$predfile" --labelfile-background=bg.tsv || exit
		fi
	done
else
	while read -r -d $'\0' predfile; do
		echo "$predfile"
		"$here"/eval.py "$predfile" --labelfile-background=bg.tsv || exit
	done < <(find "$here" -name '*.pred.pkl' -print0)
fi
