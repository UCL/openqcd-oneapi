#!/bin/bash

mkdir -p processed &> /dev/null
EXTENSION="bin"
for fn in $(find . -maxdepth 1 -name "*.$EXTENSION")
do
    BASENAME=$(basename --suffix=".$EXTENSION" $fn)
    mv $fn processed/$BASENAME-$1-$2-$3-$4.$EXTENSION
done
