# SFCLS
Sparse Fully Constrained Least Squares for spectral unmixing

It is shown in associated paper that Lp norm promotes sparsity in spectral unmixing and improves results over other sparsity promoting regularization terms. LASSO and reciprocal L_infty are also considered, although they do not have as high of performance. 

Below is a description of the file struction, instructions to use the unmixing algorithms in your own experiments, and instructions to reproduce results shown in the paper.
____________________________________________________________
File Structure

- "unmix" : main project folder, contains scripts for running experiments, files with callable 
            functions and classes, and sub-directories for input data and output results
    - "input" : sub-directory with input data
        - "endmemb_libs" :  endmember libraries with spectra and meta data
        - "kim" :   spectra and meta data with spectra, meta data and ground truth from
                    kim feely 1999 dataset and experiment
        - "mola" :  mars mola shaded relief map to overlay behind mineral maps of mars
        - "Raw_TES" : Mars Thermal Emission Spectrometer data set. Too large to push to git, can be querried from TES web tool or can be 			shared by contacting author
    - "results" : sub-directory where experiment scripts store data
        - <Year-Month-Day> : directories where raw data from experiments are sorted by date
	- "final": associated results for final experiments
        - "mineral_maps" : mineral maps generated from unmixing prediction
	- "plots" : results displayed in figures
    - *endmember_class.py* : class for storing endmember libraries
    - *mixclass.py* : class for storing mixture spectra and meta data
    - *unmix.py* : callable spectral unmixing functions
    - *synthmix.py* : callable function for generating synthetic mixtures
    - <test_XXX.py> : scripts for testing spectral unmixing functions are working as expected
    - <run_XXX.py> : scripts for running experiments
    - <display_XXX.py> : scripts for displaying / printing data

____________________________________________________________

Instructions to use spectral unmixing algorithms in your own experiments

1. Install dependencies:
	```
	pip install numpy scipy
	```
2. Place *unmix.py* file in your project folder or path

3. Import spectral unmixing function with the following line of code

	``` python
	from unmix import pNorm_unmix
	```
    - OR inftyNorm_unmix, LASSO_unmix, FCLS_unmix
    
4. Call the function with the following line of code (and additional kwargs)

	```python
	x = pNorm_unmix(A, b, *kwargs)
	```    
    - A : numpy ndarray of size MxN. each column of A is an endmember spectra
        M is number of spectral channels, N is number of endmembers
    - b : numpy ndarray of size M that represents a mixture spectra to be unmixed
    
    - returns x : numpy ndarray of size N, each element of x is the abundance
                of the endmember corresponding to the columns in A
    
    valid kwargs for each function are
    
    - lam : float64, weight of regularization (inftyNorm, LASSO, pNorm)
    - p : float64, order of p_norm recomended 0.9<p<1 (p_norm)
    - surface: list or range of surface endmember indices, according to endmember library A Only used if there are atmospheric endmembers which have different constraints/regularization. 
    - lam_atm: float64, weight of atmospheric L2 norm regularization. Defaults to 0. Only needed if there are atmospheric endmembers in library.
    - maxiter: int, maximum number of iterations of scipy optimizers
    - ftol: Tolerance of optimizer exit criteria. It is a threshold for the relative change in the objective function between iterations. When the objective function changes by an amount less than ftol, the optimization is complete.


_____________________________________________________________

Instructions to reproduce experiment:
1. clone git hub repo:
	
	```
	git clone https://github.com/johnjaniczek/SFCLS.git
	```
	
2. Install dependencies OR use provided virtual environment
	```
	pip install spectral numpy scipy pandas sklearn matplotlib feather-format h5py
	```
	
3. Run test scripts to verify unmixing algorithms are functioning:
	
	- test_FCLS.py

	- test_LASSO_unmix.py
	
	- test_inftyNorm.py
	
	- test_pNorm.py

4. Run experiment on 1000 synthetic endmember mixtures with noise standard deviation = 2.5e-4 and 2.5e-5

	- execute run_synthmix.py with hyperparameters set according to file unmix/results/final/2019-07-12_synthmix_params0.csv
	- execute run_synthmix.py with hyperparameters set according to file unmix/results/final/2019-07-12_synthmix_params1.csv
	- output is in unmix/results/<Todays Date>
	
5. Run experiment on Feely 1998 dataset (laboratory mixtures)

	- execute run_synthmix.py with hyperparameters set according to file unmix/results/final/2019-07-12_labmix_params0.csv
	- output is in unmix/results/<Todays Date>

6. Run experiment on Mars TES data

	- acquire MARS TES data from TES web tool or inquire author to share data
	- execute run_TES_raw.py with p = 0.999, lam_pnorm = 1e-2, and lam_atm=1e-2

7. Display mineral maps
	
	- execute display_TES.py (editing the "run" parameter according to the file generated by step 6)
	- output is in unmix/results/mineral_maps


    
    
