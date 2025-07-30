import pymc as pm
import numpy as np
import pandas as pd

def gen_model(observed_data):
    # Convert observed_data columns to numpy arrays
    time_surgery = observed_data['time_surgery'].to_numpy()
    metastasized_status = observed_data['metastasized_status'].to_numpy()
    
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(314)
    
    with pm.Model() as model:
        # Create pm.MutableData objects for each non-observation column
        time_surgery_data = pm.MutableData("time_surgery", time_surgery, dims="obs_id")
        metastasized_status_data = pm.MutableData("metastasized_status", metastasized_status, dims="obs_id")
        
        # Define priors for the coefficients
        beta_time = pm.Normal("beta_time", mu=0, sigma=1)
        beta_metastasized = pm.Normal("beta_metastasized", mu=0, sigma=1)
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        
        # Define the linear combination
        logit_p = intercept + beta_time * time_surgery_data + beta_metastasized * metastasized_status_data
        
        # Define the likelihood using a Bernoulli distribution
        y_obs = pm.Bernoulli("y_obs", logit_p=logit_p, observed=None, dims="obs_id")
        
        # Sample from the prior
        prior_predictive = pm.sample_prior_predictive(samples=1000, random_seed=rng2, return_inferencedata=False)
        
        return model, prior_predictive