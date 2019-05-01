# SFCLS
Sparse Fully Constrained Least Squares experiment

____________________________________________________________
Instructions to reproduce experiment:
1. clone git hub repo
	git clone https://github.com/johnjaniczek/SFCLS.git

2. Install dependencies
	pip install spectral numpy pandas sklearn cvxpy csv 

3. Run test scripts
	test_FCLS.py
	test_LASSO_unmix.py
	test_SFCLS.py

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


