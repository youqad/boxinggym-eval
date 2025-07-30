import pymc as pm
import numpy as np

def gen_model(observed_data):
    # Convert observed_data columns to numpy arrays
    group1_group2_count_diff = observed_data['group1_group2_count_diff'].to_numpy()
    group1_group2_gender_diff = observed_data['group1_group2_gender_diff'].to_numpy()
    group1_group2_age_diff = observed_data['group1_group2_age_diff'].to_numpy()
    group1_group2_social_status_diff = observed_data['group1_group2_social_status_diff'].to_numpy()
    group1_group2_fitness_diff = observed_data['group1_group2_fitness_diff'].to_numpy()
    group1_group2_species_diff = observed_data['group1_group2_species_diff'].to_numpy()
    intervention = observed_data['intervention'].to_numpy()
    choice = observed_data['choice'].to_numpy()

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(314)
    
    with pm.Model() as model:
        # Create a pm.MutableData object for each non-observation column
        group1_group2_count_diff = pm.MutableData("group1_group2_count_diff", group1_group2_count_diff, dims='obs_id')
        group1_group2_gender_diff = pm.MutableData("group1_group2_gender_diff", group1_group2_gender_diff, dims='obs_id')
        group1_group2_age_diff = pm.MutableData("group1_group2_age_diff", group1_group2_age_diff, dims='obs_id')
        group1_group2_social_status_diff = pm.MutableData("group1_group2_social_status_diff", group1_group2_social_status_diff, dims='obs_id')
        group1_group2_fitness_diff = pm.MutableData("group1_group2_fitness_diff", group1_group2_fitness_diff, dims='obs_id')
        group1_group2_species_diff = pm.MutableData("group1_group2_species_diff", group1_group2_species_diff, dims='obs_id')
        intervention = pm.MutableData("intervention", intervention, dims='obs_id')
        
        # Priors for the coefficients
        beta_count = pm.Normal("beta_count", mu=0, sigma=1)
        beta_gender = pm.Normal("beta_gender", mu=0, sigma=1)
        beta_age = pm.Normal("beta_age", mu=0, sigma=1)
        beta_social_status = pm.Normal("beta_social_status", mu=0, sigma=1)
        beta_fitness = pm.Normal("beta_fitness", mu=0, sigma=1)
        beta_species = pm.Normal("beta_species", mu=0, sigma=1)
        beta_intervention = pm.Normal("beta_intervention", mu=0, sigma=1)
        
        # Logistic regression for the probability of choosing group 2
        logit_p = (beta_count * group1_group2_count_diff +
                   beta_gender * group1_group2_gender_diff +
                   beta_age * group1_group2_age_diff +
                   beta_social_status * group1_group2_social_status_diff +
                   beta_fitness * group1_group2_fitness_diff +
                   beta_species * group1_group2_species_diff +
                   beta_intervention * intervention)
        
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        
        # Likelihood
        y_obs = pm.Bernoulli("y_obs", p=p, observed=(choice == 2).astype(int), dims="obs_id")
        
        # Sampling
        trace = pm.sample(1000, tune=500, target_accept=0.90, chains=3, cores=1, random_seed=rng1, idata_kwargs={"log_likelihood": True})
        posterior_predictive = pm.sample_posterior_predictive(trace, random_seed=rng2, return_inferencedata=False)
        
        return model, posterior_predictive, trace