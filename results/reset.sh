#!/usr/bin/env bash

# cleans all .dat files from results

# remove data files and accuracy log and create new empty log
rm nbow/q-type/*.dat
rm nbow/ag-news/*.dat
rm nbow/stk-ovflw/*.dat
rm nbow/q-type/accuracy_log.txt
rm nbow/ag-news/accuracy_log.txt
rm nbow/stk-ovflw/accuracy_log.txt
touch nbow/q-type/accuracy_log.txt
touch nbow/ag-news/accuracy_log.txt
touch nbow/stk-ovflw/accuracy_log.txt


# remove data files and accuracy log and create new empty log
rm lstm/q-type/*.dat
rm lstm/ag-news/*.dat
rm lstm/stk-ovflw/*.dat
rm lstm/q-type/accuracy_log.txt
rm lstm/ag-news/accuracy_log.txt
rm lstm/stk-ovflw/accuracy_log.txt
touch lstm/q-type/accuracy_log.txt
touch lstm/ag-news/accuracy_log.txt
touch lstm/stk-ovflw/accuracy_log.txt

# remove data files and accuracy log and create new empty log
rm tcnn/q-type/*.dat
rm tcnn/ag-news/*.dat
rm tcnn/stk-ovflw/*.dat
rm tcnn/q-type/accuracy_log.txt
rm tcnn/ag-news/accuracy_log.txt
rm tcnn/stk-ovflw/accuracy_log.txt
touch tcnn/q-type/accuracy_log.txt
touch tcnn/ag-news/accuracy_log.txt
touch tcnn/stk-ovflw/accuracy_log.txt

# remove data files and accuracy log and create new empty log
rm dcnn/q-type/*.dat
rm dcnn/ag-news/*.dat
rm dcnn/stk-ovflw/*.dat
rm dcnn/q-type/accuracy_log.txt
rm dcnn/ag-news/accuracy_log.txt
rm dcnn/stk-ovflw/accuracy_log.txt
touch dcnn/q-type/accuracy_log.txt
touch dcnn/ag-news/accuracy_log.txt
touch dcnn/stk-ovflw/accuracy_log.txt
