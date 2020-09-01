There is a lot of variability in the reporting of cases (due to several factors), and therefore we need to smooth the data before fitting.

# Rolling Average

Rolling Avg params - 
- window size
- left, right or center window
- window type

Currently rolling average is done before the data is split. Therefore, there is slight leakage of data from val to train. For rolling after splitting, dealing with endpoints becomes tricky. For that we have the following options : 
- No RA for endpoints
- Dynamic window (that reduces to 1 at the endpoint)
- Using model prediction as endpoint data input instead

# Big Jump Smoothing

