# Fast-characterization-of-optically-detected-magnetic-resonance-spectra-via-data-clustering
This repository will contain the main functions and a sample set of simulated data used in the paper "Fast characterization of optically detected magnetic resonance spectra via data clustering".

## Function Bank
- The Function_Bank.py file contains the main functions used in the paper including the custom Clustering Algorithm (CA), the data simulation function, and a couple of smaller companion functions. 
- If you plan on using any of the functions ensure that you read the function documentation first as some things were designed for the specific data used in this study. Some small modifications may need to be made depending on your use case and data format. 

## Sample Simulated Data
A small subset of the simulated datasets used in this study are included. The data is organised into files where every set in the file has a similar total number of counts. 

### Structure
- Shapes: 100 rows (datasets), 202 columns
- The first 199 columns are the bins containing the photoluminescence (PL) counts
- The last two columns contain the true peak locations
- The third last column contains the specific total number of counts for the dataset (given the probabilistic nature of the simulation, datasets total successful counts cannot be precisely chosen ahead of time)

### Use With Function Bank
The sample simulated datasets are formatted such that they can be processed by the CA out-of-the-box. These sets can be used to get a handle on how the function works. If you wish to replicate a small portion of the results using these datasets use the following parameters for CA(...):
- N_range = [1.5, 2.5, 3.75, 5.0, 15.0, 25.0]
- k_ys = 4
- MW_range = [3, 4]
- ver = 2
