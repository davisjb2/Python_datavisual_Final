#!/bin/sh
echo latex thesis
latex thesis
echo bibex thesis
bibtex thesis
echo latex thesis
latex thesis
./toc.sh
echo pdflatex thesis
pdflatex thesis
#dvipdfm thesis

