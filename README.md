# SFCLS
Experiment for promoting sparsity with full constraints
in spectral unmixing problems
____________________________________________________________
File Structure

- "unmix" : main project folder, contains scripts for running experiments, files with callable 
            functions and classes, and sub-directories for input data and output results
    - "input" : sub-directory with input data
        - "endmemb_libs" :  endmember libraries with spectra and meta data
        - "kim" :   spectra and meta data with spectra, meta data and ground truth from
                    kim feely 1999 dataset and experiment
        - "mola" :  mars mola shaded relief map to overlay behind mineral maps of mars
        - "surface_emissivity" : pre-processed surface emissivity from bandfield 2002
    - "results" : sub-directory where experiment scripts store data
        - <Year-Month-Day> : directories where raw data from experiments are sorted by date
        - "mineral_maps" : mineral maps generated from unmixing prediction
    - *endmember_class.py* : class for storing endmember libraries
    - *mixclass.py* : class for storing mixture spectra and meta data
    - *unmix.py* : callable spectral unmixing functions
    - *synthmix.py* : callable function for generating synthetic mixtures
    - <test_XXX.py> : scripts for testing spectral unmixing functions are working as expected
    - <run_XXX.py> : scripts for running experiments
    - <display_XXX.py> : scripts for displaying data

____________________________________________________________
Instructions to reproduce experiment:
1. clone git hub repo:
	
	git clone https://github.com/johnjaniczek/SFCLS.git

2. Install dependencies:
	
	pip install spectral numpy scipy pandas sklearn cvxpy matplotlib

3. Run test scripts:
	
	test_FCLS.py

	test_LASSO_unmix.py
	
	test_SFCLS.py
	
	test_p_norm.py
	
	test_delta_norm.py

4. Run experiment on 1000 synthetic endmember mixtures with noise standard deviation = 2.5e-5
	
	execute run_SFCLS.py with hyperparameters: lam = 3e-7, K = 1000, gauss_dev = 2.5e-5, min_endmemb = 0.01, thresh = 0.01
	
	execute run_LASSO.py with hyperparameters: lam = 1e-4, K = 1000, gauss_dev = 2.5e-5, min_endmemb = 0.01, thresh = 0.01
	
	execute run_FCLS.py with hyperparameters K = 1000, gauss_dev = 2.5e-5, min_endmemb = 0.01, thresh = 0.01

	output is in "results/<Todays date>"

5. Run experiment on Feely 1998 dataset (laboratory mixtures)

	execute run_realmix.py with hyperparameters: SFCLS_lam = 1e-4, LASSO_lam = 1e-4, thresh = 0.01

	output is in "results/<Todays date>"

6. Run experiment on TES pre processed surface emissivity

	execute run_TES.py with hyperparameters: SFCLS_lam = 1e-4, LASSO_lam = 1e-4

	output is in "results/<TES_unmix_raw>"

7. Display mineral maps
	
	execute display_TES.py (possibly need to edit filenames to match results produced from step 6)

_____________________________________________________________

Instructions to use spectral unmixing algorithms in your own experiments

1. Install dependencies and clone github repo per above instructions

2. Place *unmix.py* file in your project folder or path

3. Import spectral unmixing function with the following line of code

    from unmix import SFCLS_unmix
    
    also valid to import FCLS_unmix, LASSO_unmix, p_norm_unmix, delta_norm_unmix
    
4. Call the function with the following line of code (and additional kwargs)

    x = SFCLS_unmix(A, b, *kwargs)
    
    A : numpy ndarray of size MxN. each column of A is an endmember spectra
        M is number of spectral channels, N is number of endmembers
    b : numpy ndarray of size M that represents a mixture spectra to be unmixed
    
    returns x : numpy ndarray of size N, each element of x is the abundance
                of the endmember corresponding to the columns in A
    
    valid kwargs for each function are
    
    lam : float64, weight of regularization (SFCLS, LASSO, p_norm, delta_norm)
    p : float64, order of p_norm recomended 0.7<p<1 (p_norm)
    delta: float64, delta parameter in the delta_norm regularization sum(x/(x+delta))
    