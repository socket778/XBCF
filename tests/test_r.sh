#! /bin/bash
echo Building R
cd ../../
R CMD REMOVE XBCF
R CMD INSTALL XBCF
cd XBCF/tests/
echo Testing R
# Rscript test.R
