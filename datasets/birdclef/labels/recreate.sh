#!/bin/bash
here="${0%/*}"
python "$here"/recreate.py "$here"/../xml/{train,test,soundscapes} "$here"
