#!/bin/bash

# check if flatc is installed
if ! [ -x "$(command -v flatc)" ]; then
#   echo 'Error: flatc is not installed.' >&2
  echo 'Installing flatbuffers-compiler'
  sudo apt-get install flatbuffers-compiler
fi

# Generate flatbuffer files
flatc --python -o tmsg/generated schema/heatmap.fbs
