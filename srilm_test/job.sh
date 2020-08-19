#!/bin/bash
path=/home/josh/SRILM/bin/i686-m64
dtbasename=testset
#-wbdiscount
$path/ngram-count -order 2 -text dataset.txt -lm lm_train.arpa
#-debug 3 -skipoovs
#> ppl_all.txt
$path/ngram -debug 3 -lm lm_train.arpa -ppl ${dtbasename}.txt > ppl_all.txt