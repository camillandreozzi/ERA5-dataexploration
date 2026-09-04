# ERA5-CLARA P-O-C

Proof of concept:
current work is subset to a small time frame (2020-12-01 to 2020-12-05)

## Step by step:
1. read_inclara.py
2. data_fetch.py based on subset decided in read_inclara.py
3. read_inera5.py
4. data_overlap.py
5. data_preprocessing.py
6. explore_merged.py

## For the modelling
benchmark_lasso and benchmark_rf only model the mean function of OLR
in rf_stkriging the attempt is to combine a non-linear mean modelling with a space-time kriging of the residuals 