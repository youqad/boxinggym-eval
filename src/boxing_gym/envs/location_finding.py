import ast
import numpy as np
import random
import logging
from itertools import permutations

from scipy.stats import norm
import pymc as pm

from .goal import Goal
from ..agents.box_loop_helper import construct_dataframe

logger = logging.getLogger(__name__)

class DirectGoal(Goal):
    # Default env params used to compute hardcoded norm values
    _NORM_DEFAULTS = {"num_sources": 3, "dim": 2, "m": 1e-4, "alpha": 1}

    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0
        self.update_param_cache = {}
        # Recomputed with Normal(0,1) sampling to match eval distribution
        # Values from 50-trial median to reduce variance
        # IMPORTANT: Only valid for default params (num_sources=3, dim=2, m=1e-4, alpha=1)
        self.norm_mu = 176.9
        self.norm_sigma = 1247.7

        # Warn if env params differ from those used to compute hardcoded norms
        self._check_norm_validity(env)
    
    def get_system_message(self, include_prior):
        if include_prior:
            goal_description = f"""
Your goal is to be able to reliably predict the signal intensity (scalar-values) at any given {self.env.dim}-dimensional coordinate.
"""
        else:
            goal_description = f"""
Your goal is to be able to reliably predict a scalar-valued floating-point number for any given {self.env.dim}-dimensional vector.
"""
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    def get_goal_eval_question(self, include_prior):
        if self.eval_pointer+1 > len(self.eval_points):
            loc = np.random.normal(0, 1, (self.env.dim, ))
            choice = self.env.signal_intensity(loc)
            self.eval_points.append((loc, choice))
        else:
            loc, choice = self.eval_points[self.eval_pointer]
        self.eval_pointer += 1

        if include_prior:
            question = f"""
Predict the intensity at the following location: {loc}. 
When asked to answer a question about the environment, respond in the format specified below. Make assumptions about the environment and provide your best guess. Answer up to two decimal places.
Here is an example.
<thought> your thought </thought>
<answer>1</answer>
"""
            assert "location" in question
            assert "intensity" in question

        else:
            question = f"""
Predict the scalar valued quanties at the following coordinates: {loc}.
When asked to answer a question about the environment, respond in the format specified below. Make assumptions about the environment and provide your best guess. Answer up to two decimal places.
Here is an example.
<thought> your thought </thought>
<answer>1</answer>
"""
            assert "location" not in question
            assert "intensity" not in question
        return question, choice 

    def evaluate_predictions(self, predictions, measurements):
        assert len(predictions) == len(measurements), "Predictions and measurements must have the same length."

        filtered_predictions = []
        filtered_measurements = []
        for prediction, measurement in zip(predictions, measurements):
            if prediction is not None:
                prediction = prediction.strip()
                if "=" in prediction:
                    prediction = prediction.split("=")[1]
                filtered_predictions.append(float(prediction))
                filtered_measurements.append(measurement)
        se = (np.array(filtered_predictions) - np.array(filtered_measurements)) ** 2
        mse = np.mean(se)
        std_mse = np.std(se)
        return mse, std_mse

    def expected_information_gain(self, query_point, num_outer_samples=1000, num_inner_samples=10):
        existing_data = self.env.observed_data
        if tuple(existing_data) not in self.update_param_cache:
            print("updating param cache")
            # Update the prior distribution using the existing data
            with pm.Model() as model:
                # Define dimensions
                theta = pm.Normal('theta', mu=0, sigma=1, shape=(self.env.num_sources, self.env.dim))

                # Define the likelihood function
                def likelihood(xi, obs_y):
                    intensity = self.env.b
                    for k in range(self.env.num_sources):
                        distance_squared = pm.math.sum((theta[k] - xi) ** 2)
                        intensity += self.env.alpha / (self.env.m + distance_squared)
                    return pm.Normal('signals', mu=intensity, sigma=self.env.sigma, observed=obs_y)

                # Observe the existing data
                # Convert existing_data to numpy arrays for better handling
                if len(existing_data) > 0:
                    obs_loc = np.array([obs_point for obs_point, _ in existing_data])
                    obs_intensity = np.array([obs_intensity for _, obs_intensity in existing_data])

                    # Use pm.Data to allow for dynamic updating
                    obs_loc_data = pm.Data('obs_loc', obs_loc)
                    obs_intensity_data = pm.Data('obs_intensity', obs_intensity)

                    # Call the likelihood function with the observed data
                    likelihood(obs_loc_data, obs_intensity_data)

                    # Sample from the posterior distribution
                    trace = pm.sample(num_outer_samples * num_inner_samples + num_outer_samples, tune=1000, return_inferencedata=False)
                    thetas = trace['theta']
                else:
                    # sample num_outer_samples * num_inner_samples + num_outer_samples thetas
                    thetas = np.random.normal(0, 1, (num_outer_samples * num_inner_samples + num_outer_samples, self.env.num_sources, self.env.dim))
                self.update_param_cache[tuple(existing_data)] = thetas
        else:
            thetas = self.update_param_cache[tuple(existing_data)]

        random.shuffle(thetas)
        outer_thetas = thetas[:num_outer_samples]

        log_likelihoods = []
        for n, outer_theta in enumerate(outer_thetas):
            inner_thetas = thetas[num_outer_samples+n * num_inner_samples:num_outer_samples+(n + 1) * num_inner_samples]
            # Calculate the log-likelihood for the new query point only
            intensity = self.env.b
            for k in range(len(outer_theta)):
                distance_squared = np.sum((outer_theta[k] - query_point) ** 2)
                intensity += self.env.alpha / (self.env.m + distance_squared)
            sampled_obs = np.random.normal(intensity, self.env.sigma)
            log_likelihood = norm.logpdf(sampled_obs, loc=intensity, scale=self.env.sigma)

            # Calculate the marginal log-likelihood for the new query point only
            marginal_log_likelihoods = []
            for inner_theta in inner_thetas:
                intensity_inner = self.env.b
                for k in range(len(inner_theta)):
                    distance_squared = np.sum((inner_theta[k] - query_point) ** 2)
                    intensity_inner += self.env.alpha / (self.env.m + distance_squared)
                marginal_log_likelihood = norm.logpdf(sampled_obs, loc=intensity_inner, scale=self.env.sigma)
                marginal_log_likelihoods.append(marginal_log_likelihood)

            # Use the log-sum-exp trick for numerical stability
            max_log = np.max(marginal_log_likelihoods)
            log_marginal_likelihood = max_log + np.log(np.mean(np.exp(np.array(marginal_log_likelihoods) - max_log)))
            log_likelihoods.append(log_likelihood - log_marginal_likelihood)

        # Calculate the EIG for the new query point
        eig_value = np.mean(log_likelihoods)
        return eig_value

    def _check_norm_validity(self, env):
        """Warn if env params differ from those used to compute hardcoded norms."""
        mismatches = []
        for param, default in self._NORM_DEFAULTS.items():
            actual = getattr(env, param, default)
            if actual != default:
                mismatches.append(f"{param}={actual} (expected {default})")
        if mismatches:
            logger.warning(
                f"location_finding env params differ from norm defaults: {', '.join(mismatches)}. "
                f"Hardcoded norm values (mu={self.norm_mu}, sigma={self.norm_sigma}) may be invalid. "
                "Consider calling get_norm_factors() to recompute."
            )

    def get_norm_factors(self):
        N = 100
        predictions = []
        measurements = []
        for _ in range(N):
            self.env.reset()
            # Match eval distribution: Normal(0,1) instead of Uniform(-2,2)
            loc = np.random.normal(0, 1, (self.env.dim,))
            # Use noiseless signal_intensity to match eval (not noisy step())
            signal = self.env.signal_intensity(loc)
            # No >100 filter - eval doesn't filter either
            measurements.append(signal)
        mu_pred = np.mean(measurements)
        predictions = [str(mu_pred.tolist()) for _ in range(len(measurements))]
        mean_mse, std_mse = self.evaluate_predictions(predictions, measurements)
        return mean_mse, std_mse

class DirectGoalNaive(DirectGoal):
    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0
    
    def get_system_message(self, include_prior):
        goal_description = "Your goal is to conduct experiments and explain the environment to the user so that they can achieve their goal."

        if include_prior:
            goal_description = f"""
The goal of the user is to be able to reliably predict the signal intensity (scalar-values) at any given {self.env.dim}-dimensional coordinate.
"""
        else:
            goal_description = f"""
The goal of the user is to be able to reliably predict a scalar-valued floating-point number for any given {self.env.dim}-dimensional vector.
"""
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    def get_naive_system_message(self, include_prior):
        if include_prior:
            goal_description = f"""
Your goal is to be able to reliably predict the signal intensity (scalar-values) at any given {self.env.dim}-dimensional coordinate.
"""
        else:
            goal_description = f"""
Your goal is to be able to reliably predict a scalar-valued floating-point number for any given {self.env.dim}-dimensional vector.
"""
        format_instructions = """
You will be provided with a set of inputs to this environment and will be tasked with predicting the output for each input.
You must give number values. You may also think before providing your predictions.
You must respond with a single number! 
If you do not do so, I will be penalized one million dollars and it will be a complete disaster.
Here is an example:
<thought>your thought</thought>
<answer>1</answer>"""
        description = goal_description + '\n' + format_instructions
        description += "Here is what you know about the environment:\n"
        return description    
    
    def get_comm_prompt(self, include_prior, com_limit=300, use_ppl=False, str_prob_prog=None, params_summary_str=None):    
        extra_prompt = ""
        if include_prior:
            extra_prompt = """
Importantly, they need to be able to predict the intensity without knowing the source locations.
Do not give an extremely detailed protocol for making predictions but rather general heuristic guidelines that do not require detailed calculations0.
Emphasize how you can make predictions without knowing the source locations.
"""
        description = f"""
Assume that the user has no prior knowledge, and will not be able to run any experiments before making predictions. 
They will make predictions based solely on your explanation, so provide as much detail as possible. Do not directly provide your experiments and observations.
Limit your explanation to {com_limit} words.
{extra_prompt}

"""
        if use_ppl:
            description += "To make your explanation clearer and more informative, look at the statistical model (written in pymc) designed by a colleague for the experimental data and the inferred parameters. \n"
            description += f"Here is the statistical model. \n {str_prob_prog} \n"
            description += f"Here are the inferred params. \n {params_summary_str} \n"
            description += "Don't literally describe the model verbatim but use it to conceptually motivate your explanation."
            description += "The agent will not be able to use the model explicitly but having a conceptual understanding will be beneficial."
        return description
    

class SourceGoal(Goal):
    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0
        self.update_param_cache = {}
        self.norm_mu = 1.57
        self.norm_sigma = 1.15385
    
    def get_system_message(self, include_prior):
        if include_prior:
            goal_description = f"""
Your goal is to be able to predict the locations ({self.env.dim}-dimensional coordinates) of {self.env.num_sources} signal sources.
"""
        else:
            goal_description = f"""
Your goal is to be able to predict {self.env.num_sources} sets of {self.env.dim}-dimensional latent parameters that govern the system's scalar outputs.
"""
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    def get_goal_eval_question(self, include_prior):
        true_theta = self.env.true_theta
        if include_prior:
            question = f"""
Please predict {self.env.num_sources} {self.env.dim}-dimensional coordinates of the signal sources.
Format your prediction as a list of {self.env.num_sources} lists of {self.env.dim} values. Answer up to two decimal places.
Here is an example prediction: '[{', '.join(['[' + ', '.join(['1' for _ in range(self.env.dim)]) + ']' for _ in range(self.env.num_sources)])}]'
"""
        else:
            question = f"""
Please predict {self.env.num_sources} sets of {self.env.dim}-dimensional latent parameters.
Format your prediction as a list of {self.env.num_sources} lists of {self.env.dim} values. Answer up to two decimal places.
Here is an example prediction: '[{', '.join(['[' + ', '.join(['1' for _ in range(self.env.dim)]) + ']' for _ in range(self.env.num_sources)])}]'
"""
        return question, true_theta 
    
    def evaluate_predictions(self, predictions, measurements):

        assert len(predictions) == len(measurements), "Predictions and measurements must have the same length."

        def _parse_prediction(prediction):
            if isinstance(prediction, (list, tuple)):
                parsed = list(prediction)
            elif isinstance(prediction, str):
                candidate = prediction.strip()
                if not candidate.startswith("[") and "[" in candidate and "]" in candidate:
                    candidate = candidate[candidate.find("["):candidate.rfind("]") + 1]
                try:
                    parsed = ast.literal_eval(candidate)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"Invalid prediction format: {prediction}") from exc
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            if not isinstance(parsed, (list, tuple)):
                raise ValueError(f"Prediction must be a list: {parsed}")
            parsed = list(parsed)
            parsed = [list(item) if isinstance(item, tuple) else item for item in parsed]
            return parsed

        predictions = [_parse_prediction(x) for x in predictions]

        def permuted_mse(prediction, true_theta):
            assert isinstance(prediction, list)
            estimated_locations = np.array(prediction)
            if estimated_locations.shape != true_theta.shape:
                raise ValueError("The shape of the estimated locations must match the shape of the true locations.")

            lowest_mse = float('inf')
            for permuted_estimated in permutations(estimated_locations):
                se = (self.env.true_theta - np.array(permuted_estimated)) ** 2
                mse = np.mean(se)

                if mse < lowest_mse:
                    lowest_mse = mse

            return lowest_mse
        
        mses = [permuted_mse(predictions[i], measurements[i]) for i in range(len(predictions))]
        return np.mean(mses), np.std(mses)

    def expected_information_gain(self, query_point, num_outer_samples=1000, num_inner_samples=10):
        existing_data = self.env.observed_data
        if tuple(existing_data) not in self.update_param_cache:
            print("updating param cache")
            # Update the prior distribution using the existing data
            with pm.Model() as model:
                # Define dimensions
                theta = pm.Normal('theta', mu=0, sigma=1, shape=(self.env.num_sources, self.env.dim))

                # Define the likelihood function
                def likelihood(xi, obs_y):
                    intensity = self.env.b
                    for k in range(self.env.num_sources):
                        distance_squared = pm.math.sum((theta[k] - xi) ** 2)
                        intensity += self.env.alpha / (self.env.m + distance_squared)
                    return pm.Normal('signals', mu=intensity, sigma=self.env.sigma, observed=obs_y)

                # Observe the existing data
                # Convert existing_data to numpy arrays for better handling
                if len(existing_data) > 0:
                    obs_loc = np.array([obs_point for obs_point, _ in existing_data])
                    obs_intensity = np.array([obs_intensity for _, obs_intensity in existing_data])

                    # Use pm.Data to allow for dynamic updating
                    obs_loc_data = pm.Data('obs_loc', obs_loc)
                    obs_intensity_data = pm.Data('obs_intensity', obs_intensity)

                    # Call the likelihood function with the observed data
                    likelihood(obs_loc_data, obs_intensity_data)

                    # Sample from the posterior distribution
                    trace = pm.sample(num_outer_samples * num_inner_samples + num_outer_samples, tune=1000, return_inferencedata=False)
                    thetas = trace['theta']
                else:
                    # sample num_outer_samples * num_inner_samples + num_outer_samples thetas
                    thetas = np.random.normal(0, 1, (num_outer_samples * num_inner_samples + num_outer_samples, self.env.num_sources, self.env.dim))
                self.update_param_cache[tuple(existing_data)] = thetas
        else:
            thetas = self.update_param_cache[tuple(existing_data)]

        random.shuffle(thetas)
        outer_thetas = thetas[:num_outer_samples]

        log_likelihoods = []
        for n, outer_theta in enumerate(outer_thetas):
            inner_thetas = thetas[num_outer_samples+n * num_inner_samples:num_outer_samples+(n + 1) * num_inner_samples]
            # Calculate the log-likelihood for the new query point only
            intensity = self.env.b
            for k in range(len(outer_theta)):
                distance_squared = np.sum((outer_theta[k] - query_point) ** 2)
                intensity += self.env.alpha / (self.env.m + distance_squared)
            sampled_obs = np.random.normal(intensity, self.env.sigma)
            log_likelihood = norm.logpdf(sampled_obs, loc=intensity, scale=self.env.sigma)

            # Calculate the marginal log-likelihood for the new query point only
            marginal_log_likelihoods = []
            for inner_theta in inner_thetas:
                intensity_inner = self.env.b
                for k in range(len(inner_theta)):
                    distance_squared = np.sum((inner_theta[k] - query_point) ** 2)
                    intensity_inner += self.env.alpha / (self.env.m + distance_squared)
                marginal_log_likelihood = norm.logpdf(sampled_obs, loc=intensity_inner, scale=self.env.sigma)
                marginal_log_likelihoods.append(marginal_log_likelihood)

            # Use the log-sum-exp trick for numerical stability
            max_log = np.max(marginal_log_likelihoods)
            log_marginal_likelihood = max_log + np.log(np.mean(np.exp(np.array(marginal_log_likelihoods) - max_log)))
            log_likelihoods.append(log_likelihood - log_marginal_likelihood)

        # Calculate the EIG for the new query point
        eig_value = np.mean(log_likelihoods)
        return eig_value
    
    def get_norm_factors(self):
        N = 1000000
        predictions = []
        measurements = []
        errs = []
        for _ in range(N):
            self.env.reset()
            # sample sources from prior
            sources = np.random.normal(0, 1, (self.env.num_sources, self.env.dim))
            theta = self.env.true_theta
            err = np.mean((sources - theta) ** 2)
            errs.append(err)
            measurements.append(theta)
        mu = np.zeros_like(theta)
        predictions = [str(mu.tolist()) for _ in range(N)]
        std = np.std(errs)
        mean_mse, _ = self.evaluate_predictions(predictions, measurements)
        return mean_mse, std

class Signal:
    def __init__(self, num_sources=3, dim=2, b=1e-1, m=1e-4, alpha=1, sigma=0.5):
        # Initialize constants
        self.b = b
        self.m = m
        self.alpha = alpha
        self.sigma = sigma

        # Initialize variables
        self.num_sources = num_sources
        self.dim = dim  # Number of dimensions
        self.reset()
        self.env_name = "location_finding"
    
    def reset(self):
        self.observed_data = []
        # Randomly generate source locations with normal distribution
        self.true_theta = np.random.normal(0, 1, (self.num_sources, self.dim))

    def generate_system_message(self, include_prior, goal_description):
        if include_prior:
            PRIOR = f"""The intensity at any point in the grid is determined by the superposition of signals from {self.num_sources} sources.
Each source emits a signal of identical strength. 
Note that the location of the sources is unknown to us!
Make observations by specifying a single point where you want to measure the signal in a length-{self.dim} list of floating-point numbers enclosed by double brackets. 
"""
            message= f"""{PRIOR}
{goal_description}
Here is an example:
<thought> your thought </thought>
<observe>[0.5, 1]</observe>
When asked to answer a question about the environement, respond in the format specified in the question.
Example:
<thought> your thought </thought>
<answer> your answer </answer>
"""
        else:
            NO_PRIOR = f"""Each point on the grid (a {self.dim}-dimensional vector) has an associated floating-point number. 
Make observations by specifying a single point where you want to observe in a {self.dim} list of floating-point numbers. 
"""
            message = f"""{NO_PRIOR}
{goal_description}
Here is an example:
<thought> your thought </thought>
<observe>[0.5, 1]</observe>
When asked to answer a question about the environement, respond in the format specified in the question.
Example:
<thought> your thought </thought>
<answer> your answer </answer>
"""
        return message
    
    def sample_random_input(self):
        # Use Normal(0,1) to match: (1) source locations, (2) eval distribution
        # Previously Uniform(-2,2) caused distribution mismatch
        return np.random.normal(0, 1, (self.dim,))

    def signal_intensity(self, xi):
        total_intensity = self.b
        for k in range(len(self.true_theta)):
            distance_squared = np.sum((self.true_theta[k] - xi) ** 2)
            total_intensity += self.alpha / (self.m + distance_squared)
        return total_intensity

    def step(self, xi):
        intensity = self.signal_intensity(xi)
        noisy_intensity = np.random.normal(intensity, self.sigma)
        return noisy_intensity
    
    def validate_input(self, input_string):
        try:
            data = input_string.strip()
            data = input_string.strip("[]")
            data = [item.strip() for item in data.split(",")]
            data = [float(item) for item in data] 
        except:
            return "Error: Input must be a list of floating-point numbers."
        if len(data) != self.dim:
            return "Error: Input must be a list of length equal to the dimension of the environment."
        return np.array(data)

    def run_experiment(self, input_string):
        query = self.validate_input(input_string)
        if type(query) == str:
            return query, False
        obs = self.step(query)
        # convert obs to float from np array
        obs = float(obs)
        # round to 2 decimal places
        obs = round(obs, 2)
        # convert query to list from np array
        query = tuple(query.tolist())
        self.observed_data.append((query, obs))
        return obs, True

    def get_data(self):
        return self.observed_data

    def get_df(self):
        '''
            Construct dataframe used for Box's Loop
        '''
        self.df = construct_dataframe(self)
    
    def get_description(self):
        if self.include_prior:
            return f"""The intensity at any point in the grid is determined by the superposition of signals from {self.num_sources} sources.
Each source emits a signal of identical strength. 
Note that the location of the sources is unknown to us!
"""
        else:
            return f"""Each point on the grid (a {self.dim}-dimensional vector) has an associated floating-point number. 
Make observations by specifying a single point where you want to observe in a {self.dim} list of floating-point numbers. 
"""

    def describe_data_columns(self):
        return self.format_column_description()

    def get_ordered_column_names(self):
        if self.include_prior:
            return ["Coordinate_1", "Coordinate_2", "Signal_Intensity"]
        else:
            return ["Input_1", "Input_2", "Output"]
    
    def get_ordered_features(self):
        return self.get_ordered_column_names()[:-1] 

    def format_column_description(self):
        '''
            Crucial make sure these descriptions are consistent with the ordered column names
        '''
        if self.include_prior:
            return ("The observations are: \n -Signal_Intensity: signal strength at the location \n"
                    "The input values are \n -Coordinate_1: 1st dimension coordinate \n -Coordinate_2: 2nd dimension coordinate \n"
                    "Use the values of the input values to help you model the observations. ")
        else:
            return ("The observations are: \n -Output \n"
                    "The input values are \n -Input_1 \n -Input_2 \n"
                    "Use the values of the input values to help you model the observations. ")

if __name__ == "__main__":
    env = Signal(3, 2)
    goal = DirectGoal(env)
    print(goal.get_norm_factors())
    # input_string = "[0.5, 0.5]"
    # parsed_input = goal.env.validate_input(input_string)
    # print(parsed_input)
    # print(env.run_experiment(input_string))
    # # inp = env.sample_random_input()
    # # # get eig
    # # print(inp)
    # # print(goal.expected_information_gain(inp))
    # # input_string = "[0.5, 1]"
    # # print(goal.env.observed_data)
    # # run a few more experiments
    # # input_string = "[1, 1]"
    # # # get eig
    # # print(goal.env.run_experiment(input_string))
    
    # # input_string = "[0, 0]"
    # # print(goal.env.run_experiment(input_string))
    # inputs = [
    #         "[0.5, 0.5]",
    #         "[0.75, 0.5]",
    #         "[0.5, 0.75]",
    #         "[0.5, 0.25]",
    #         "[0.25, 0.5]",
    #         "[0.75, 0.25]",
    #         "[0.25, 0.75]",
    #         "[0.75, 0.75]",
    #         "[0.25, 0.25]",
    #         "[0.9, 0.1]"
    # ]
    # inputs_arr = [goal.env.validate_input(inp) for inp in inputs]
    # print(inputs_arr)
    # for inp in inputs:
    #     parsed_input = goal.env.validate_input(inp)
    #     print(parsed_input)
    #     print(goal.expected_information_gain(parsed_input))
    #     print(goal.env.run_experiment(inp))

    #     print(goal.expected_information_gain(goal.env.validate_input(inp)))
    #     print(goal.env.run_experiment(inp))

    # # x_range = np.linspace(-2, 2, 10)
    # # y_range = np.linspace(-2, 2, 10)
    # # eigs = []

    # for x in x_range:
    #     for y in y_range:
    #         query_point = np.array([x, y])
    #         print(f"Calculating EIG for query point: {query_point}")
    #         eig = goal.expected_information_gain(query_point)
    #         eigs.append(eig)
    #         print(f"EIG: {eig}")

    # # Reshape the EIG values into a 2D grid
    # eigs_grid = np.array(eigs).reshape((len(x_range), len(y_range)))

    # # Create a heatmap of the EIG values
    # plt.figure(figsize=(8, 6))
    # plt.imshow(eigs_grid, cmap='viridis', origin='lower', extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]])
    # plt.colorbar(label='EIG')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('EIG Heatmap')
    # plt.show()
    # # test run experiment
