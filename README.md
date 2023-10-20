# Contents

This folder contains the following files

1. Jupyter notebooks for drug effectiveness study, unstructured covariates and GWAS experiments with changing datasets+changing base propensity models
2. changing\_causalprop\_exp.py which compares the performance of our method while the proportion of causal SNPs is changed. 
3. computational\_times.py which compares the computational resources required by Naive Bayes and Logistic Regression
4. Folder deep-gwas-vae contains the code to generate simulated GWAS datasets
5. Folder simulated_gwas contains a few simulated datasets we generate using the code in deep-gwas-vae

# How to run the experiments:

You can use the file base_env.yml to create conda environment for our experiments by using the command

conda env create -f base_env.yml

## Experiments on drug effectiveness study and unstructured covariates

The Jupyter notebooks can be run directly. 

## Generating simulated GWAS datasets:

cd gwas_simulator/src/data

python gwas\_sim.py --config=../../configs/dataset/gwas\_sp0\_3.yaml --save\_file=../../datasets/simulated\_gwas/sample\_run.pt

Above command uses the gwas\_sp0\_3.yaml file to extract the GWAS simulation configuration and stores the generated dataset in the folder gwas\_simulator/datasets/simulated\_gwas/sample\_run.pt. You can design your own yaml file or supply the required settings (i.e., causal proportion, number of SNPs, random seed, etc.) through command line arguments. 


## Comparison of GWAS computational times:

Please make sure that the required GWAS datasets are generated using the above step and the appropriate filename is supplied on line 164 of computational_times.py.


Naive Bayes:

python computational\_times.py --numsnps=1 --eval_length=100

Logistic Regression:

python computational\_times.py --numsnps=1 --eval_length=100 --classifier=1


## Comparing GWAS setups:
Please make sure that the required GWAS datasets are generated and the appropriate filename is supplied on line 167 of changing_causalprop_exp.py.

The Jupyter notebooks can be run directly after the dataset is generated and stored in the folder simulated\_datasets. 



### Example run of changing\_causalprop\_exp.py (spatial dataset)

python changing\_causalprop\_exp.py --causalprop=1 --eval_length=100 --classifier=1

Here, causalprop argument can be used to index within any causal SNP proportion in the list [0.01, 0.02, 0.05, 0.1]. Similarly, the classifier argument can be used to index into the list [GaussianNB(), LogisticRegression(), MLPClassifier(), RandomForestClassifier(), AdaBoostClassifier(), MLPClassifier()] as base propensity classifiers. 

To change the simulated dataset being used, you can directly change the name on line 167 of the file. 


