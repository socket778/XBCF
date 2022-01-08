#! /bin/bash
echo Building R
cd ../../
R CMD REMOVE XBCF
R CMD INSTALL XBCF
cd XBCF/tests/
echo Testing R
Rscript visualize_gp.R
