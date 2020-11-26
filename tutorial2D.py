import numpy as np
import datetime
import os
import json
import yaml
import math
import time
import torch
import gpytorch
import matplotlib.pyplot as plt
#excursion package imports
import excursion
import excursion.testcases.fast_2D as testcase_details
import excursion.plotting.twodim as plots
from excursion import init_gp
from excursion import ExcursionSetEstimator



## Step 1: Load
for func in testcase_details.true_functions:
    ix,iy = testcase_details.plot_meshgrid
    X = testcase_details.X
    cont = excursion.utils.values2mesh(func(X), testcase_details.rangedef, testcase_details.invalid_region)
    levels = plt.contourf(ix,iy,cont)
    plt.contour(ix,iy,cont, levels = [0.], color='white')
    
levels.collections[0].set_label('excursion set at t='+str(testcase_details.thresholds.item()))
cbar = plt.colorbar(levels)
cbar.ax.set_ylabel('f(x,y)')
#approximate level set for plot purposes
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D_fast.py')
plt.legend(loc=0)
plt.savefig('plots/test_fast2D.png')

#load algorithm options
file = open('excursion/testcases/algorithms/algorithm_specs.yaml', "r")
algorithm_opts = yaml.safe_load(file)
print('algorithm_opts =')
print(json.dumps(algorithm_opts, indent=4))
## Step 2: Init the Gaussian Proces
#We initialize the gaussian process and likelihood according to `algorithm_opts['init_type']`
if (torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model, likelihood = init_gp(testcase_details, \
                            algorithm_opts, \
                            algorithm_opts["ninit"], \
                            device)

# X_init and y_init are random points so that we can fit the GP posterior wrt these points
# fit hyperparameters
model.train()
likelihood.train()
excursion.fit_hyperparams(model,likelihood)
# plot the fitting
plots.plot_GP_init(plt, model, testcase_details, device=device, dtype=torch.float64)
## Step 3: Init the ExcursionEstimator
estimator = ExcursionSetEstimator(testcase_details, \
                                  algorithm_opts, \
                                  model, \
                                  likelihood, \
                                  device)
## Step 4:  Smart grid search (active learning)
n_updates = algorithm_opts['nupdates']
#save results in folder
timestampStr = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S") + "/"
outputfolder = "examples/results/" +  timestampStr
os.makedirs(outputfolder)

for iteration in range(1, n_updates+1):    
    #take a step forward: find new points to in testcase_details.true_functions:
    estimator.step(testcase_details, algorithm_opts, model, likelihood)
    
    #plot the new status
    estimator.plot_status(
            testcase_details, algorithm_opts, model, estimator.acq_values, outputfolder)
    
    #update the posterior of the GP wrt the new points
    model = estimator.update_posterior(testcase_details, algorithm_opts, model, likelihood)
    
    #get diagnostics (accuracy metric)
    estimator.get_diagnostics(testcase_details, model, likelihood)
    
#save results
estimator.print_results(outputfolder, testcase_details, algorithm_opts)
