#!/bin/bash

mkdir tmp
cd tmp


# Create svhn annotations
echo "#### Create IIS annotations ####"
git clone https://github.com/penny4860/svhn-voc-annotation-format

mkdir -p ../tests/dataset/iis/train/anns
mkdir -p ../tests/dataset/iis/test/anns
mv svhn-voc-annotation-format/annotation/train/* ../tests/dataset/iis/train/anns
mv svhn-voc-annotation-format/annotation/test/* ../tests/dataset/iis/test/anns


# Create svhn images
echo "#### Create IIS images ####"
curl http://ufldl.stanford.edu/housenumbers/train.tar.gz -o train.tar.gz
tar -xzvf train.tar.gz
mv train ../tests/dataset/iis/train/imgs

curl http://ufldl.stanford.edu/housenumbers/test.tar.gz -o test.tar.gz
tar -xzvf test.tar.gz
mv test ../tests/dataset/iis/test/imgs


# Remove tmp directory
cd ..
rm -rf tmp/