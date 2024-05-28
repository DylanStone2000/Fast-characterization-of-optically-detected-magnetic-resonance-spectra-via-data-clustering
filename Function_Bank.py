# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:45:30 2024

@author: Dylan G. Stone 

Python Version: 3.10
Spyder: 5.5.1

This file contains the main functions used in our paper "Fast characterization 
of optically detected magnetic resonance spectra via data clustering", namely, 
ODMR simulation and our custom Clustering Algorithm. 
"""

import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def Sim_ODMR(MW_range, x01, x02, width1, width2, n_tries, success_prob1, 
             success_prob2, n_bins, plot=False):
    """
    This function takes a handful of parameters and generates a simulated OMDR
    spectra. 
    
    Parameters
    ----------
    MW_range : Array or list of two float64's
        This should be a list of two values, the beginning and the end of the MW
        that will be used.
    x01 : float64
        The position of the first peak in the spectrum in MHz.
    x02 : float64
        The position of the second peak in the spectrum in MHz.
    width1 : float64
        The width of the first peak in the spectrum in MHz.
    width2 : float64
        The width of the second peak in the spectrum in MHz.
    n_tries : int32
        The number of tries to allow for a count in each bin in the MW range.
    success_prob1 : float64
        The contrast for the first peak (diff between wings and peak in 
        normalized spectrum). 
    success_prob2 : float64
        The contrast for the second peak (diff between wings and peak in 
        normalized spectrum). 
    n_bins : int32
        The number of bins to divide the MW range into.
    plot : bool, optional
        A flag that determines if the simulated dataset is plotted. 
        The default is False.

    Returns
    -------
    ODMR_data : array of float64
        An array of the counts in each of the bins in the MW range
    N : int32
        The total number of counts/bin
    """     
    
    def simulate_data(resonance_value, range_start, range_end, num_points, 
                      num_tries, success_probability_at_resonance, width):
        
        # array to store successful events per bin
        successful_events_per_bin = np.zeros(num_points)  
        
        for i, value in enumerate(np.linspace(range_start, range_end, num_points)):
            
            # prob of success at a given frequency "value" 
            success_prob = success_probability_at_resonance / \
                (1 + ((value - resonance_value) / width) ** 2)
           
            # number of success if you try "num_tries" times (binomial distr.)
            successes = np.random.binomial(num_tries, success_prob, 1)[0]
            successful_events_per_bin[i] = successes
        
        return successful_events_per_bin
    
    success_per_bin1 = simulate_data(x01, MW_range[0], MW_range[1], n_bins, 
                                     n_tries, success_prob1, width1)
    success_per_bin2 = simulate_data(x02, MW_range[0], MW_range[1], n_bins, 
                                     n_tries, success_prob2, width2)
    
    N = (sum(success_per_bin1) + sum(success_per_bin2)) / n_bins
    
    ODMR_data = np.zeros(n_bins)
    for i in range(len(ODMR_data)):
        ODMR_data[i]=(n_tries-(success_per_bin1[i]+success_per_bin2[i]))/n_tries
    
    if plot == True:
        freq_axis = np.linspace(MW_range[0], MW_range[1], n_bins)
        
        plt.figure(f'x01,2:{x01},{x02}; width1,2:{width1},{width2}; Prob1,2:{success_prob1},{success_prob2}')
        plt.clf()
        plt.plot(freq_axis, ODMR_data, 'o', alpha=0.6)
        plt.xlabel('Microwave Frequency (MHz)')
        plt.ylabel('Normalized Count')

    return ODMR_data, N


def contains(a, b, threshold_percentage):
    
    """
    The purpose of this function is to check if a set is partially or fully
    contained within another set. This is used in custom clustering algorithm.
    
    Parameters
    ----------
    a : array of float64
        The set you want to determine whether it is a subset of b.
    b : array of float64
        The set you want to determine if it contains a.
    threshold_percentage : float64
        A decimal value which is used as the threshold for the partial subset.
        Ex: if you want to check if 80% of a is within b, this value would be 0.8.

    Returns
    -------
    bool
        Returns True if a is partially/fully a subset if b, False otherwise.

    """
    
    count = np.in1d(a, b).sum()
    
    # Calculate the percentage
    percent = (count / len(a)) * 100
    
    # Check if the percentage is greater than or equal to the threshold
    return percent >= threshold_percentage
 
    
def CA(data_path, N_range, k_ys, MW_range, n_sets=0, max_iter=300,
              metrics=True, export=True, plot=False, ver=2):
    
    """
    A cusotm Clustering Algorithm (CA) based on Kmeans clustering made to fit 
    ODMR data. The algorithm first clusters the data using the vertical values 
    (visually: row clustering). The points from the lowest cluster are then 
    cluster again, this time specifically into two clusters to identify the 
    two peaks. The centroids of said clusters are then the peak predictions.
    There are two additional conditions in place to check for specific cases 
    which would otherwise fail to make accurate predictions (see Supplimentary 
    Information)
    
    Required Data Structure
    -----------------------
    Structure: 
    CSV files where each row is a single dataset, and each column is a single 
    bin of counts, with the last two columns containing the peak locations. 
    
    **NOTE: The function is currently set up assumg the peak locations are 
        scaled based on index location between 0 and 198, and are then scaled
        to be between MW_range min and max before clustering. The function 
        Scale() (below) can be used to change your dataset if needed. 
    
    Our Study:
    For this study, the files had shapes (number of sets, 201) (ver=1) or 
    (number of sets, 202) (ver=2). The first 199 columns contained bin counts, 
    and the last two contained the peak locations. For ver=2 data (simulated), 
    the third last column contained the average counts/bin in the simulated 
    data.
    
    Optimal Use:
    The function was designed to process many datasets in sequence in groups 
    based on some similar characteristic, and as a result requires datasets with 
    carefully chosen names (more below). For instance, in this study we had 
    several CSV files of multiple datasets, where each file contained sets with 
    approximately a certain number of average counts/bin. Therefore, the files 
    were named based on the number of average counts/bin. 
    
    Parameters
    ----------
    data_path : string
        The path where the data is being stored.
        NOTE: this path must have a {} contained in it that allows the code
            to grab the right dataset based on inserting a value from N_range
            Ex: "./Synthetic/Counts_per_Bin={}_sim.csv"
    N_range : list or array of int32
        A list/array of numbers for the amount of average counts/bin in a dataset. 
        This is used to import multiple datasets. 
        Note: this doesn't have to be averages counts/bin, this can be easily 
        substituted for any list of values that are used to name datasets.
    k_ys : int32 or 1D array/list/dataframe of int32s
        The number of clusters for the y-axis grouping. If just a single int32
        is recieved, that valued is used for all N's. If a list is passed each 
        value of k_ys will be used for each value of N_range. 
        NOTE: if k_ys is a list, k_ys and N_range must have the same length.
    MW_range : list or array of two values
            The start and end values (in order) of the MW range the data spans. 
    n_sets : int32, optional
        The number of rows of data (i.e. number of datasets) to import. The 
        default is 0, which imports ALL rows of the csv.
    max_iter : int32, optional
        The maximum iterations for the KMeans function. The default is 300.
    metrics : bool, optional
        A flag that determines whether all metrics are returned. If True, 
        accuracy, loop performance, and elapsed times are returned. If False, 
        only accuracy is returned. The default is True.
    export : bool, optional
        A flag that determines if the metrics are exported to a csv file.
        If True, the files are exported. The default is True.
    exception : bool, optional
        A flag that determines if the number of exceptions handled is stored
        in the loop_perf dataframe. The default is True.
    plot : bool, optional
        A flag that determines if a prediction plot is created. If True, 
        a plot is created. The default is False.
    ver : int32
        Used to determine where the range of features are in the inputted table
        based on the two different versions of data simulation for ODMR. The 
        default is 1 to ensure old files run without editing. 

    Returns
    -------
    pandas dataframe(s)
        Dataframe(s) containing various metrics per N.
        accuracy: contains two columns per CSV, the accuracy of each peak 
            prediction.
        loop_performance: containes two columns ver CSV, the average caccuracy 
            of both peaks and all predictions (single value) and the STDEV of 
            said average.
        elapsed_time: contains one column per CSV, the time said loop took to 
            complete, as well as a final column for the total time taken in 
            the function. 

    """

    # start timer for entire function run
    start_time = time.time()

    if type(k_ys) == int or type(k_ys) == float:
        k_ys = np.ones(len(N_range), dtype=int)*k_ys

    elif len(k_ys) != len(N_range):
        print('WARNING: k_ys needs to have the same length as N_range')
        return
    
    if ver != 1 and ver != 2:
        print('WARNING: version number can only be 1 or 2')
        return

    # determine boundaries of x data
    if ver == 2:
        stop = -3
    else:
        stop = -2
        
    # Performance metrics
    elapsed_time = pd.DataFrame()
    loop_perf = pd.DataFrame()
    accuracy = pd.DataFrame()

    for k in range(len(N_range)):
        # start timer for specific loop run
        loop_time = time.time()
        
        N = N_range[k]
        k_y = k_ys[k]
        
        # Importing and scaling data
            # if n_sets is specified use it, otherwise use the entire dataset
        if n_sets != 0:
            data = pd.read_csv(data_path.format(N), nrows=n_sets) 
        else:
            data = pd.read_csv(data_path.format(N)) 
        #data.shape[1]-3 = 198 -> the maximum index for the features
        data['f1'] = Scale(data['f1'], 0, 198, MW_range[0], MW_range[1])
        data['f2'] = Scale(data['f2'], 0, 198, MW_range[0], MW_range[1])

        # array containing MW values
        MW = np.linspace(MW_range[0], MW_range[1], data.shape[1]+stop)
        # actual peak locations
        y = data.iloc[:, -2:].values
        # counts
        x = data.iloc[:, :stop].values
        
        # Initializing lists
        # Row clustering
        models = []
        labels = []
        cluster_centers = []
        # column clustering (two groups)
        peaks = []
        acc = []
        
        # For each set -> Row Cluster
        for i in range(data.shape[0]):
            # Create ROW clustering model (cluster based on vertical values)
                # fit data needs to be a column
            model = KMeans(n_clusters=k_y, init='k-means++', n_init=10, max_iter=max_iter, algorithm='lloyd').fit(x[i].reshape(-1,1))
            models.append(model)
            labels.append(model.labels_) # group labels
            cluster_centers.append(model.cluster_centers_) # cluster centroids    
            
        # For each set -> Column Cluster
        for i in range(data.shape[0]): 
            indice_options = []  # will store the indeces for the lowest, two lowest, etc. clusters
            model_x_options = [] # will store the different models based on the indice options
            
            lowest = True  # a flag to determine if this is the lowest (working) cluster pair
            
            # Lowest Cluster (rest down in loop below) 
                # sort cluster_centers, make list of the indeces
                # gets a list of the indices for the lowest clusters
            cluster_indices = np.argsort(cluster_centers[i][:,0])[:0+1]
            indice_options.append(np.concatenate(([np.where(labels[i]==indices)[0] \
                                                   for indices in cluster_indices])))
            # Create COLUMN clustering model (cluster based on horizontal values) using MW subset
            # create subset of MW using indice_options for fit
            try:
                model_x = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, 
                                 algorithm='lloyd').fit(MW[indice_options[0]].reshape(-1,1))
                model_x_options.append(model_x)
            
            # If clustering fails
            except ValueError:
                model_x_options.append([])
            
            
            # Loop through all-1 "lowest" options (until prediction is made) to 
            # compare the changes in widths vs distances
            for j in range(k_y-1): # k_y options means k_y-1 comparisons
                
                # Loop through "lowest" cluster configs (lowest, two lowest, etc.; first one done above) 
                    # sort cluster_centers, make list of the indeces, grab all of them up to j+1
                    # gets a list of the indices for the jth lowest clusters
                    # must be (j+1) instead of j since first iteration is done outside the loop
                cluster_indices = np.argsort(cluster_centers[i][:,0])[:(j+1)+1]
                indice_options.append(np.concatenate(([np.where(labels[i]==indices)[0] \
                                                       for indices in cluster_indices])))
                # Create COLUMN clustering model (cluster based on horizontal values) using MW subset
                # create subset of MW using indice_options for fit
                try:
                    model_x = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, 
                                     algorithm='lloyd').fit(MW[indice_options[(j+1)]].reshape(-1,1))
                    model_x_options.append(model_x)
                
                # If clustering fails
                except ValueError:
                    model_x_options.append([])
                
            
                if type(model_x_options[j]) != list:
                    
                    # The "current" model option (j)
                    cluster1 = np.where(model_x_options[j].labels_==0)[0]
                    cluster2 = np.where(model_x_options[j].labels_==1)[0]
                    width1 = np.abs(np.max(MW[cluster1])-np.min(MW[cluster1]))
                    width2 = np.abs(np.max(MW[cluster1])-np.min(MW[cluster1]))
                    width_current = (width1 + width2)/2 
                    
                    centroids = model_x_options[j].cluster_centers_[:,0]
                    cluster_distance_current = np.abs(centroids[0]-centroids[1])
                    
                    # The "next" model option (j+1)
                    cluster1 = np.where(model_x_options[j+1].labels_==0)[0]
                    cluster2 = np.where(model_x_options[j+1].labels_==1)[0]
                    width1 = np.abs(np.max(MW[cluster1])-np.min(MW[cluster1]))
                    width2 = np.abs(np.max(MW[cluster1])-np.min(MW[cluster1]))
                    width_next = (width1 + width2)/2 
                    
                    centroids = model_x_options[j+1].cluster_centers_[:,0]
                    cluster_distance_next = np.abs(centroids[0]-centroids[1])
                    
                    comparison = ((cluster_distance_next/cluster_distance_current)/(width_next/width_current))
                    
                    if comparison <= 1.0:
                        # if the change is <=1.0, then j must already include a second peak
                        if lowest == False:
                            # Figure out which new subcluster contains the previous two
                            
                            # The previous model option (j-1)
                            prev_clust1 = np.where(model_x_options[j-1].labels_==0)[0] # one of the previous subclusters
                            prev_clust2 = np.where(model_x_options[j-1].labels_==1)[0] # one of the previous subclusters
                            
                            # The current model option (j)
                            cluster1 = np.where(model_x_options[j].labels_==0)[0]
                            cluster2 = np.where(model_x_options[j].labels_==1)[0]
                            
                            # check which new subcluster contains "all" (80%) the points from the previous
                            if contains(prev_clust1, cluster1, 80) == True:
                                # replace this subcluster with both of the previous 
                                cluster1 = np.concatenate((prev_clust1, prev_clust2))
                                
                                # recluster using the additional layer for only one of them
                                model_x_new = KMeans(n_clusters=2, init='k-means++', n_init=10, 
                                                  max_iter=300, algorithm='lloyd').fit(MW[indice_options[j]][np.concatenate((cluster1,cluster2))].reshape(-1,1))
                                peaks.append(model_x_new.cluster_centers_)
                                break
                                
                            elif contains(prev_clust1, cluster2, 80) == True:
                                # replace this subcluster with both of the previous 
                                cluster2 = np.concatenate((prev_clust1, prev_clust2))
                                
                                # recluster using the additional layer for only one of them
                                model_x_new = KMeans(n_clusters=2, init='k-means++', n_init=10, 
                                                  max_iter=300, algorithm='lloyd').fit(MW[indice_options[j]][np.concatenate((cluster1,cluster2))].reshape(-1,1))
                                peaks.append(model_x_new.cluster_centers_)
                                break
                                
                            else:
                                peaks.append(model_x_options[j-1].cluster_centers_)
                                break
                                                
                        if lowest == True:
                            peaks.append(model_x_options[j].cluster_centers_)
                            # layer_used.append(j+1)
                        break
                                    
                    lowest = False
                
                # If peaks are never selected (and we're on the last j)
                if j == (k_y-1)-1:
                    for m in range(len(model_x_options)):
                        if type(model_x_options[m]) != list: 
                            peaks.append(model_x_options[m].cluster_centers_)
                            break
                    if type(model_x_options[m]) == list:
                        peaks.append(np.array([[np.nan], [np.nan]]))
            
            # find the accuracy of the model
                # np.sort ensures that we are comparing the correct peaks
            acc.append(np.abs(np.sort(peaks[i].reshape(1,-1)) - np.sort(y[i])))
        
        # array version to help with dataframe below    
        # stack list elements to create 2D array 
        acc_array = np.vstack(acc)
                    
        # keep track of metrics for each loop
        accuracy[f'{N}, acc, f1'] = acc_array[:,0]
        accuracy[f'{N}, acc, f2'] = acc_array[:,1] 
        loop_perf[f'{N}, acc'] = [np.nanmean(acc_array)]
        loop_perf[f'{N}, std'] = [np.nanstd(acc_array)]
        elapsed_time[f'{N}'] = [time.time() - loop_time]
        
        if plot == True:
            for i in range(data.shape[0]):
                plt.figure(f'Set {i}: Peaks (N={N})')
                plt.clf()
                plt.scatter(MW, x[i].reshape(199), c=labels[i])
                plt.vlines(peaks[i][0], x[i].min(), x[i].max(), label='f1 & f2')
                plt.vlines(peaks[i][1], x[i].min(), x[i].max())
                plt.xlabel('MW (GHz)')
                plt.ylabel('Normalized Counts')

        print(f'Loop: {N}\nTime Elapsed: {time.time()-loop_time}')

    elapsed_time['Total'] = [time.time()-start_time]        

    if export == True:
        
        # get date and time to append to file names
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        
        # Export metrics
        accuracy.to_csv(f'accuracy_KMeans-{now}.csv', index=False)
        loop_perf.to_csv(f'loop_performance_KMeans-{now}.csv', index=False)
        elapsed_time.to_csv(f'time_elapsed_KMeans-{now}.csv', index=False)
    
    if metrics == True:
        return accuracy, loop_perf, elapsed_time
    
    else:
        return accuracy
    

def Scale(x, x_min, x_max, new_min, new_max):    
    """
    The purpose of this function is to scale an array of data to be between a
    given min and max. Useful for switching between raw, normalised, and index 
    representations. 

    Parameters
    ----------
    x : array of float64
        Array of values you wish to scale.
    x_min : float64
        The minimum value that x COULD have.
            Ex: If you are looking at frequencies from 3 to 4GHz, 3 is the min
            even if all the values in the array x are larger than 3. 
    x_max : float64
        The maximum value that x COULD have.
            Ex: If you are looking at frequencies from 3 to 4GHz, 4 is the max
            even if all the values in the array x are smaller than 4.
    new_min : float64
        The new minimum value x COULD have after being scaled.
            Ex: If you are looking at frequencies from 3 to 4GHz, 3 is the old
            lower bound and new_min is the new lower bound, even if no values
            in x end up being new_min
    new_max : float64
        The new maximum value x COULD have after being scaled.
            Ex: If you are looking at frequencies from 3 to 4GHz, 4 is the old
            upper bound and new_max is the new upper bound, even if no values
            in x end up being new_max.

    Returns
    -------
    scaled_x : array of float64
        The array of scaled data.

    """
    
    # Scale to desired range
    scaled_x = (new_max - new_min)*(x - x_min)/(x_max - x_min) + new_min
    
    return scaled_x