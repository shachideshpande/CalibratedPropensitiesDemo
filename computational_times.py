import gc
import sys
from math import exp, log
import pdb
import pandas as pd
import torch
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from scipy.special import expit
import argparse
from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression,  LogisticRegressionCV, LinearRegression, Ridge, ElasticNet
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit, learning_curve, validation_curve
from sklearn.model_selection import train_test_split
from sklearn import clone

from numpy import linalg as LA

import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pdb
import os.path
from numpy import array
from numpy_sugar.linalg import economic_qs_linear
from glimix_core.lmm import LMM

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.rcParams['figure.figsize'] = 10, 8

np.seterr(divide='ignore', invalid='ignore')
np.random.seed(42)


total_plain_time = np.zeros(3)
total_calib_time = np.zeros(3)


#####

def record_start():
	start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
	return start_time, start_resources

def update_total_time(start_time, start_resources, iscalib=False):

	end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()
	global total_plain_time, total_calib_time
	if(iscalib): # real, sys, user
		total_calib_time[0]+=end_time - start_time
		total_calib_time[1]+=end_resources.ru_stime - start_resources.ru_stime
		total_calib_time[2]+=end_resources.ru_utime - start_resources.ru_utime
	else: # real, sys, user
		total_plain_time[0]+=end_time - start_time
		total_plain_time[1]+=end_resources.ru_stime - start_resources.ru_stime
		total_plain_time[2]+=end_resources.ru_utime - start_resources.ru_utime


def ece(y_true, y_pred, n_bins=10, title='text'):
    x_axis = np.arange(0, 1.1, (1.0)/n_bins)
    y_axis = np.zeros(x_axis.shape)
    x_axis_new = np.zeros(x_axis.shape)
    bin_count = np.zeros(x_axis.shape)
    zero_indices = []
    N = len(y_true)
    score = 0
    for i, x in enumerate(x_axis):
        if(i==0):
            continue
        bin_outputs = y_true[np.logical_and(y_pred<=x, y_pred>x_axis[i-1])]
        bin_preds = y_pred[np.logical_and(y_pred<=x, y_pred>x_axis[i-1])]
        
            
        if(len(bin_outputs)>0):
            y_axis[i]=bin_outputs.mean()
            avg_pred = bin_preds.mean()
            x_axis_new[i] = avg_pred
        else:
            y_axis[i]=0
            avg_pred = 0
            x_axis_new[i] = x
            zero_indices += [i] 
        
        
        bin_count[i] = len(bin_outputs)
        score+=abs(y_axis[i]-avg_pred)*((1.0*len(bin_outputs))/N)
    return score




    
    
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='GWAS with propensities',
	                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--classifier', type=int, default=0)
	# proportion of SNPs that are causal in simulated dataset
	parser.add_argument('--causalprop', type=int, default=0)
	# total number of SNPs in simulated dataset
	parser.add_argument('--numsnps', type=int, default=0)
	# eval-length determines the number of SNPs starting from first position for which you compute ATE
	parser.add_argument('--eval_length', type=int, default=1)
	args = parser.parse_args()
	EVAL_LENGTH = args.eval_length
	plot_diagnostics = False
	run_baselines=False
	epsilon = 1e-2
	# comparing naive bayes and LR for computational times
	classifiers = [
	GaussianNB(),
	LogisticRegression(max_iter = 100000, C=0.8),
	]
	datasets=[]
	# these are the set of seeds for which simulated datasets were generated
	# this was to ensure reproducibility
	seed_set = [0, 12345, 54321, 10001, 100001] #
	# when we increase numsnps, the computational benefit of calibrated naive bayes is more visible
	num_snp_set = [100, 1000] #
	causal_prop_set = [0.01, 0.02, 0.05, 0.1] # 

	# creating subsets
	classifiers = [classifiers[args.classifier]]
	num_snp_set =  [num_snp_set[args.numsnps]]
	causal_prop_set = [causal_prop_set[args.causalprop]]

	exp_reps = len(seed_set)

	for classifier_id, classifier in enumerate(classifiers):
		for numsnps in num_snp_set:
			for causalprop in causal_prop_set:
			# Results

				naive_est_marginal_betas = torch.zeros((exp_reps, numsnps))
				plain_iptw_est_marginal_betas = torch.zeros((exp_reps, numsnps))
				calib_iptw_est_marginal_betas = torch.zeros((exp_reps, numsnps))
				plain_aipw_est_marginal_betas = torch.zeros((exp_reps, numsnps))
				calib_aipw_est_marginal_betas = torch.zeros((exp_reps, numsnps))

				beta_pca = torch.zeros((exp_reps, numsnps))
				beta_fa = torch.zeros((exp_reps, numsnps))
				beta_lmm = torch.zeros((exp_reps, numsnps))
				beta_true = np.zeros((exp_reps, numsnps))
				beta_true_marginal = np.zeros((exp_reps, numsnps))
				cal_scores = torch.zeros((exp_reps, numsnps, 2))
				cal_scores_train = torch.zeros((exp_reps, numsnps, 2))


				for dataset_idx, seed in enumerate(seed_set):
					nextfile = "simulated_gwas/gwas_sp_real_"+str(seed)+"_"+str(numsnps)+"_"+str(causalprop)+'.pt'
					
					print(nextfile)
					if(os.path.isfile(nextfile)):
						dataset = nextfile

						print("Working with Dataset "+dataset)

						simulated_data = torch.load(dataset)


						N = simulated_data['snps'].shape[1]
						# M = 1000
						M = simulated_data['snps'].shape[0]
						G = simulated_data['snps'][:int(0.8*M)]
						Y = simulated_data['outcomes'][:int(0.8*M)]
						Z = simulated_data['clusters'][:int(0.8*M)]
						lambdas = simulated_data['true_lambdas'][:int(0.8*M)]


						G_test = simulated_data['snps'][int(0.8*M):]
						Y_test = simulated_data['outcomes'][int(0.8*M):]
						Z_test = simulated_data['clusters'][int(0.8*M):]

						beta_true[dataset_idx] = betas = simulated_data['true_betas']

						pc = PCA(n_components=10)
						pc_snps = StandardScaler().fit_transform(pc.fit_transform(G))
						pc_snps_test = StandardScaler().fit_transform(pc.transform(G_test))

						QS = economic_qs_linear(G)
						fa = FactorAnalysis(n_components=10)
						fa_snps = fa.fit_transform(G)
						fa_snps = StandardScaler().fit_transform(fa_snps)
						# iterate through snp vector sequentially
						fa_snps_test = StandardScaler().fit_transform(fa.transform(G_test))

						for snp_index in range((EVAL_LENGTH)): #


							print("Reading SNP "+str(snp_index))


							T = G[:, snp_index]
							T_test = G_test[:, snp_index]
							X = np.concatenate((G[:, :snp_index], G[:, snp_index+1:]), axis=1)

							X_test = np.concatenate((G_test[:, :snp_index], G_test[:, snp_index+1:]), axis=1)
							G_true = np.concatenate((T.reshape(-1,1), lambdas.reshape(-1,1)), axis=1)
							beta_true_marginal[dataset_idx][snp_index] = (Ridge().fit(G_true, Y).coef_)[0]

							if(run_baselines):

								G_pca = np.concatenate((T.reshape(-1,1), pc_snps), axis=1)
								beta_pca[dataset_idx][snp_index] = (Ridge().fit(G_pca, Y).coef_)[0]

								
								G_pca = np.concatenate((T_test.reshape(-1,1), pc_snps_test), axis=1)

								y = Y
								lmm = LMM(y, T, QS)
								lmm.fit(verbose=False)
								beta_lmm[dataset_idx][snp_index] = lmm.beta[0]
								G_fa = np.concatenate((T.reshape(-1,1), fa_snps), axis=1)
								beta_fa[dataset_idx][snp_index] = (Ridge().fit(G_fa, Y).coef_)[0]


								G_fa = np.concatenate((T_test.reshape(-1,1), fa_snps_test), axis=1)




	                
							reg1 = Ridge().fit(X[T==1], Y[T==1])
							reg0 = Ridge().fit(X[T==0], Y[T==0])


							naive_est_marginal_betas[dataset_idx][snp_index] = (reg1.predict(X) - reg0.predict(X)).mean() # assuming linear model

							start_time, start_resources= record_start()
							prop_model = clone(classifier).fit(X, T)

							prop_scores = np.clip(prop_model.predict_proba(X)[:, 1], a_min = epsilon, a_max = 1-epsilon) # 
							prop_scores_test = np.clip(prop_model.predict_proba(X_test)[:, 1], a_min = epsilon, a_max = 1-epsilon)
							update_total_time(start_time, start_resources)


							cal_scores[dataset_idx][snp_index][0] = ece(T_test, prop_model.predict_proba(X_test)[:, 1], title='Before calibration [Test]')
							cal_scores_train[dataset_idx][snp_index][0] = ece(T, prop_model.predict_proba(X)[:, 1], title='Before calibration [Train]')


							plain_iptw_est_marginal_betas[dataset_idx][snp_index] = ((T/prop_scores)*Y - ((1-T)/(1-prop_scores))*Y).mean()



							plain_aipw_est_marginal_betas[dataset_idx][snp_index] = plain_iptw_est_marginal_betas[dataset_idx][snp_index] + ((prop_scores - T)*reg0.predict(X)/(1 - prop_scores) + (prop_scores - T)*reg1.predict(X)/(prop_scores)).mean()

	                
							start_time, start_resources = record_start()

							prop_model_calib = CalibratedClassifierCV(clone(classifier).fit(X, T), method='isotonic', ensemble=True, cv='prefit') 

							

							prop_model_calib.fit(X, T)


							prop_scores = np.clip(prop_model_calib.predict_proba(X)[:, 1], a_min = epsilon, a_max = 1-epsilon)

							prop_scores_test = np.clip(prop_model_calib.predict_proba(X_test)[:, 1], a_min = epsilon, a_max = 1-epsilon)

							update_total_time(start_time, start_resources, iscalib=True)

							
							cal_scores[dataset_idx][snp_index][1] = ece(T_test, prop_model_calib.predict_proba(X_test)[:, 1], title='After calibration [Test]')
							cal_scores_train[dataset_idx][snp_index][1] = ece(T, prop_model_calib.predict_proba(X)[:, 1], title='After calibration [Train]')
							
							calib_iptw_est_marginal_betas[dataset_idx][snp_index] = ((T/prop_scores)*Y - ((1-T)/(1-prop_scores))*Y).mean()


							calib_aipw_est_marginal_betas[dataset_idx][snp_index] = calib_iptw_est_marginal_betas[dataset_idx][snp_index] + ((prop_scores - T)*reg0.predict(X)/(1 - prop_scores) + (prop_scores - T)*reg1.predict(X)/(prop_scores)).mean()



				EXP_REPS = len(calib_iptw_est_marginal_betas)
				dataset = "gwas_sp_real_"+str(numsnps)+"_"+str(causalprop)
				result_norms = torch.zeros(EXP_REPS, 12)
				for i in range(EXP_REPS):

					result_norms[i][0] = (LA.norm(calib_iptw_est_marginal_betas[i][:EVAL_LENGTH] - beta_true_marginal[i][:EVAL_LENGTH]))
					result_norms[i][1] = (LA.norm(plain_iptw_est_marginal_betas[i][:EVAL_LENGTH] - beta_true_marginal[i][:EVAL_LENGTH]))
					result_norms[i][2] = (LA.norm(calib_aipw_est_marginal_betas[i][:EVAL_LENGTH] - beta_true_marginal[i][:EVAL_LENGTH]))
					result_norms[i][3] = (LA.norm(plain_aipw_est_marginal_betas[i][:EVAL_LENGTH] - beta_true_marginal[i][:EVAL_LENGTH]))

					result_norms[i][4] = (LA.norm(beta_pca[i][:EVAL_LENGTH] - beta_true_marginal[i][:EVAL_LENGTH]))
					result_norms[i][5] = (LA.norm(beta_fa[i][:EVAL_LENGTH] - beta_true_marginal[i][:EVAL_LENGTH]))
					result_norms[i][6] = (LA.norm(beta_lmm[i][:EVAL_LENGTH] - beta_true_marginal[i][:EVAL_LENGTH]))
					result_norms[i][7] = (LA.norm(naive_est_marginal_betas[i][:EVAL_LENGTH] - beta_true_marginal[i][:EVAL_LENGTH]))
					result_norms[i][8] = (cal_scores_train[i][:, 0]).mean(axis=0)
					result_norms[i][9] = (cal_scores_train[i][:, 0]).std(axis=0)/np.sqrt(len(cal_scores[i]))
					result_norms[i][10] = (cal_scores_train[i][:, 1]).mean(axis=0)
					result_norms[i][11] = (cal_scores_train[i][:, 1]).std(axis=0)/np.sqrt(len(cal_scores[i]))
				
				torch.set_printoptions(precision=5, sci_mode=False)
				print("printing result")
				print(result_norms)
				print(result_norms.mean(axis=0))
				print(result_norms.std(axis=0)/np.sqrt(len(result_norms)))
				

				print(("Time required", total_plain_time, total_calib_time))
				
	    



