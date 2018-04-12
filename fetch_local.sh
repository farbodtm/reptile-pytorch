#!/bin/bash
#
# Fetch Mini-ImageNet from https://github.com/openai/supervised-reptile.
#

imagenet=/media/robotvision2/H/vijay_imagenet/train/

set -e

mkdir tmp
trap 'rm -r tmp' EXIT

if [ ! -d data ]; then
    mkdir data
fi

if [ ! -d data/miniimagenet ]; then
    mkdir tmp/miniimagenet
    for subset in train test val; do
        mkdir "tmp/miniimagenet/$subset"
        echo "Fetching Mini-ImageNet $subset set ..."
        for csv in $(ls metadata/miniimagenet/$subset); do
            echo "Fetching wnid: ${csv%.csv}"
            dst_dir="tmp/miniimagenet/$subset/${csv%.csv}"
            src_dir="$imagenet/${csv%.csv}"
            mkdir "$dst_dir"
            for entry in $(cat metadata/miniimagenet/$subset/$csv); do
                name=$(echo "$entry" | cut -f 1 -d ,)
                cp "$src_dir/$name" "$dst_dir/"
            done
            wait
        done
    done
    mv tmp/miniimagenet data/miniimagenet
fi
