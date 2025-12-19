import random
import re

import numpy as np
import pymc as pm
import litellm
from scipy.stats import norm

from .goal import Goal
from ..agents.box_loop_helper import construct_dataframe
from ..agents.model_config import get_model_config
from ..agents.litellm_utils import call_llm_sync

class DirectEmotionPrediction(Goal):
    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0
        self.update_params_cache = {}
        self.norm_mu, self.norm_sigma = (1.58508525, 0.7237143937677741)
    
    def get_system_message(self, include_prior):
        if include_prior:
            goal_description = "Your goal is to predict how the participant thinks the player feels after each spin of the wheel."
        else:
            goal_description = "Your goal is to predict the output values (y1-y8) based on the input features (x1-x7)."
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    def get_goal_eval_question(self, include_prior):
        if self.eval_pointer + 1 > len(self.eval_points):
            # sample new interesting point
            prize_upper = 100
            prize_lower = 0
            prizes = np.random.uniform(prize_lower, prize_upper, 3)
            probs = np.random.dirichlet(np.ones(3))
            # Use the realized prize value (not an index) as the 'win' feature.
            win_idx = int(np.random.choice([0, 1, 2]))
            win = float(prizes[win_idx])
            _, emotion_values = self.env.step(prizes, probs, win)
            self.eval_points.append((prizes, probs, win, emotion_values))
        else:
            prizes, probs, win, emotion_values = self.eval_points[self.eval_pointer]
        self.eval_pointer += 1
        if include_prior:
            question = f"""\nQuestion:The wheel has three possible outcomes with the following probabilities:
{prizes[0]:.1f}: {probs[0]:.2f}
{prizes[1]:.1f}: {probs[1]:.2f}
{prizes[2]:.1f}: {probs[2]:.2f}
The player has spun the wheel and landed on {win}.
How would the participant predict the player's emotions after this spin?"""
            format_instructions = """Provide values for the emotions in the following format on a scale of 1 (low)-9 (high):
Happiness: {happiness}/9
Sadness: {sadness}/9
Anger: {anger}/9
Surprise: {surprise}/9
Fear: {fear}/9
Disgust: {disgust}/9
Contentment: {contentment}/9
Disappointment: {disappointment}/9"""
        else:
            question = f"""\nQuestion:Given the following input features:
x1={prizes[0]:.1f}, x2={prizes[1]:.1f}, x3={prizes[2]:.1f}
x4={probs[0]:.2f}, x5={probs[1]:.2f}, x6={probs[2]:.2f}
x7={win:.1f} (the realized outcome value).
Predict the 8 output values."""
            format_instructions = """Provide values for the outputs in the following format on a scale of 1 (low)-9 (high):
y1: {y1}/9
y2: {y2}/9
y3: {y3}/9
y4: {y4}/9
y5: {y5}/9
y6: {y6}/9
y7: {y7}/9
y8: {y8}/9"""
        question += "\n" + format_instructions
        return question, emotion_values
    
    def evaluate_predictions(self, predictions, measurements):
        # parse the response for the emotion values
        parsed_response = []
        for response in predictions:
            try:
                if self.env.include_prior:
                    # Try strict regex first
                    match = re.search(r'Happiness: (.*?)/9\nSadness: (.*?)/9\nAnger: (.*?)/9\nSurprise: (.*?)/9\nFear: (.*?)/9\nDisgust: (.*?)/9\nContentment: (.*?)/9\nDisappointment: (.*?)/9', response)
                    if match:
                        response = [float(match.group(i)) for i in range(1, 9)]
                    else:
                        # Fallback: extract all numbers and hope we get 8 values
                        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*9', response)
                        if len(numbers) >= 8:
                            response = [float(n) for n in numbers[:8]]
                        else:
                            # Default to mid-scale (5.0) for unparsable responses
                            response = [5.0] * 8
                else:
                    # Try strict regex first
                    match = re.search(r'y1: (.*?)/9\ny2: (.*?)/9\ny3: (.*?)/9\ny4: (.*?)/9\ny5: (.*?)/9\ny6: (.*?)/9\ny7: (.*?)/9\ny8: (.*?)/9', response)
                    if match:
                        response = [float(match.group(i)) for i in range(1, 9)]
                    else:
                        # Fallback: extract all numbers and hope we get 8 values
                        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*9', response)
                        if len(numbers) >= 8:
                            response = [float(n) for n in numbers[:8]]
                        else:
                            # Default to mid-scale (5.0) for unparsable responses
                            response = [5.0] * 8
            except Exception:
                # Final fallback: treat as worst prediction (mid-scale)
                response = [5.0] * 8
            parsed_response.append(response)
        # convert measurements to list
        gts = []
        for measurement in measurements:
            gt = [measurement[key] for key in ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'contentment', 'disappointment']]
            gts.append(gt)
        # calculate the error
        error = []
        for i in range(len(parsed_response)):
            # calculate the error for each emotion
            error.append(np.mean([abs(parsed_response[i][j] - gts[i][j]) for j in range(8)]))
        mean = np.mean(error)
        std = np.std(error)
        return mean, std
    
    def expected_information_gain(self, query_point, num_outer_samples=100, num_inner_samples=10):
        prizes, probs, win = query_point
        existing_data = self.env.observed_data
        if tuple(existing_data) not in self.update_params_cache:
            with pm.Model() as model:
                # Priors for regression coefficients
                beta_win = pm.Normal('beta_wini', mu=[0.04, -0.025, -0.005, 0.02, 0, -0.005, 0.03, -0.03], sigma=0.01, shape=8)
                beta_PE = pm.Normal('beta_PEi', mu=[0.035, -0.02, -0.025, 0.01, 0, -0.02, 0.02, -0.045], sigma=0.01, shape=8)
                beta_absPE = pm.Normal('beta_absPEi', mu=[-0.015, 0.025, 0.025, 0.035, 0.005, 0.02, -0.01, 0.02], sigma=0.01, shape=8)

                # Priors for regression intercepts
                alpha = pm.Normal('alphai', mu=[4.5, 3.5, 1.7, 3.3, 1.4, 1.8, 3.7, 2], sigma=0.1, shape=8)

                # Priors for observation noises
                sigma = pm.HalfNormal('sigmai', sigma=1, shape=8)
                def likelihood(obs_prizes0, obs_prizes1, obs_prizes2,
                              obs_probs0, obs_probs1, obs_probs2 , obs_win, 
                              obs_happiness, obs_sadness, obs_anger, obs_surprise, obs_fear, obs_disgust, obs_contentment, obs_disappointment):
                    expected = obs_probs0 * obs_prizes0 + obs_probs1 * obs_prizes1 + obs_probs2 * obs_prizes2
                    PE = obs_win - expected
                    absPE = abs(PE)
                    # print shapes of all the inputs
                    mean0 = alpha[0] + beta_win[0] * obs_win + beta_PE[0] * PE + beta_absPE[0] * absPE
                    happiness = pm.Normal('happinessi', mu=mean0, sigma=sigma[0], observed=obs_happiness)
                    mean1 = alpha[1] + beta_win[1] * obs_win + beta_PE[1] * PE + beta_absPE[1] * absPE
                    sadness = pm.Normal('sadnessi', mu=mean1, sigma=sigma[1], observed=obs_sadness)
                    mean2 = alpha[2] + beta_win[2] * obs_win + beta_PE[2] * PE + beta_absPE[2] * absPE
                    anger = pm.Normal('angeri', mu=mean2, sigma=sigma[2], observed=obs_anger)
                    mean3 = alpha[3] + beta_win[3] * obs_win + beta_PE[3] * PE + beta_absPE[3] * absPE
                    surprise = pm.Normal('surprisei', mu=mean3, sigma=sigma[3], observed=obs_surprise)
                    mean4 = alpha[4] + beta_win[4] * obs_win + beta_PE[4] * PE + beta_absPE[4] * absPE
                    fear = pm.Normal('feari', mu=mean4, sigma=sigma[4], observed=obs_fear)
                    mean5 = alpha[5] + beta_win[5] * obs_win + beta_PE[5] * PE + beta_absPE[5] * absPE
                    disgust = pm.Normal('disgusti', mu=mean5, sigma=sigma[5], observed=obs_disgust)
                    mean6 = alpha[6] + beta_win[6] * obs_win + beta_PE[6] * PE + beta_absPE[6] * absPE
                    contentment = pm.Normal('contentmenti', mu=mean6, sigma=sigma[6], observed=obs_contentment)
                    mean7 = alpha[7] + beta_win[7] * obs_win + beta_PE[7] * PE + beta_absPE[7] * absPE
                    disappointment = pm.Normal('disappointmenti', mu=mean7, sigma=sigma[7], observed=obs_disappointment)
                    return happiness, sadness, anger, surprise, fear, disgust, contentment, disappointment 
                    

                if len(existing_data) > 0:
                    obs_prizes0 = pm.Data('obs_prizes0', [data[0][0] for data in existing_data]) 
                    obs_prizes1 = pm.Data('obs_prizes1', [data[0][1] for data in existing_data])
                    obs_prizes2 = pm.Data('obs_prizes2', [data[0][2] for data in existing_data])
                    obs_probs0 = pm.Data('obs_probs0', [data[1][0] for data in existing_data])
                    obs_probs1 = pm.Data('obs_probs1', [data[1][1] for data in existing_data])
                    obs_probs2 = pm.Data('obs_probs2', [data[1][2] for data in existing_data])
                    obs_win = pm.Data('obs_win', [data[2] for data in existing_data])
                    obs_happiness = pm.Data('obs_happiness', [data[3][0] for data in existing_data])
                    obs_sadness = pm.Data('obs_sadness', [data[3][1] for data in existing_data])
                    obs_anger = pm.Data('obs_anger', [data[3][2] for data in existing_data])
                    obs_surprise = pm.Data('obs_surprise', [data[3][3] for data in existing_data])
                    obs_fear = pm.Data('obs_fear', [data[3][4] for data in existing_data])
                    obs_disgust = pm.Data('obs_disgust', [data[3][5] for data in existing_data])
                    obs_contentment = pm.Data('obs_contentment', [data[3][6] for data in existing_data])
                    obs_disappointment = pm.Data('obs_disappointment', [data[3][7] for data in existing_data])
                    likelihood(obs_prizes0, obs_prizes1, obs_prizes2, obs_probs0, obs_probs1, obs_probs2, obs_win, obs_happiness, obs_sadness, obs_anger, obs_surprise, obs_fear, obs_disgust, obs_contentment, obs_disappointment)
                trace = pm.sample(num_outer_samples*num_inner_samples+num_outer_samples, tune=1000, return_inferencedata=False, chains=2, cores=2, target_accept=0.95)
            beta_wins = trace['beta_wini']
            beta_PEs = trace['beta_PEi']
            beta_absPEs = trace['beta_absPEi']
            alphas = trace['alphai']
            sigmas = trace['sigmai']
            shuffling = list(zip(beta_wins, beta_PEs, beta_absPEs, alphas, sigmas))
            random.shuffle(shuffling)
            beta_wins, beta_PEs, beta_absPEs, alphas, sigmas = zip(*shuffling)
            self.update_params_cache[tuple(existing_data)] = (beta_wins, beta_PEs, beta_absPEs, alphas, sigmas)
        else:
            beta_wins, beta_PEs, beta_absPEs, alphas, sigmas = self.update_params_cache[tuple(existing_data)]
        
        outer_beta_wins = beta_wins[:num_outer_samples]
        outer_beta_PEs = beta_PEs[:num_outer_samples]
        outer_beta_absPEs = beta_absPEs[:num_outer_samples]
        outer_alphas = alphas[:num_outer_samples]
        outer_sigmas = sigmas[:num_outer_samples]
        
        log_likelihoods = []
        for n, (beta_win, beta_PE, beta_absPE, alpha, sigma) in enumerate(zip(outer_beta_wins, outer_beta_PEs, outer_beta_absPEs, outer_alphas, outer_sigmas)):
            with pm.Model() as model:
                mean = alpha + beta_win * win + beta_PE * (win - np.dot(probs, prizes)) + beta_absPE * abs(win - np.dot(probs, prizes))
                sampled_emotions = np.random.normal(mean, sigma)
                prob_emotions = norm.pdf(sampled_emotions, mean, sigma)
                log_likelihood = np.log(prob_emotions+0.001)
                inner_beta_wins = beta_wins[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
                inner_beta_PEs = beta_PEs[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
                inner_beta_absPEs = beta_absPEs[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
                inner_alphas = alphas[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
                inner_sigmas = sigmas[num_outer_samples+n*num_inner_samples:num_outer_samples+(n+1)*num_inner_samples]
                marginal_log_likelihoods = []
                for inner_n, (beta_win, beta_PE, beta_absPE, alpha, sigma) in enumerate(zip(inner_beta_wins, inner_beta_PEs, inner_beta_absPEs, inner_alphas, inner_sigmas)):
                    mean = alpha + beta_win * win + beta_PE * (win - np.dot(probs, prizes)) + beta_absPE * abs(win - np.dot(probs, prizes))
                    prob_emotions = norm.pdf(sampled_emotions, mean, sigma)
                    marginal_log_likelihood = np.log(prob_emotions+0.001)
                    marginal_log_likelihoods.append(marginal_log_likelihood)
            max_log = np.max(marginal_log_likelihoods)    
            log_marginal_likelihood = max_log + np.log(np.mean(np.exp(np.array(marginal_log_likelihoods) - max_log)))
            log_likelihoods.append(log_likelihood - log_marginal_likelihood)
        eig_value = np.mean(log_likelihoods)
        return eig_value

    def get_norm_factors(self):
        N = 1000    
        measurements = []
        import logging
        import tqdm
        logger = logging.getLogger("pymc")
        logger.setLevel(logging.WARNING)
        for i in tqdm.tqdm(range(N)):
            if i % 100 == 0:
                self.env.reset()
            prizes, probs, win = self.env.sample_random_input()
            out, emotions = self.env.step(prizes, probs, win)
            emo_values = [emotions[key] for key in ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'contentment', 'disappointment']]
            measurements.append(emo_values)
        mu_happiness = np.mean([m[0] for m in measurements])
        mu_sadness = np.mean([m[1] for m in measurements])
        mu_anger = np.mean([m[2] for m in measurements])
        mu_surprise = np.mean([m[3] for m in measurements])
        mu_fear = np.mean([m[4] for m in measurements])
        mu_disgust = np.mean([m[5] for m in measurements])
        mu_contentment = np.mean([m[6] for m in measurements])
        mu_disappointment = np.mean([m[7] for m in measurements])
        mu_answer = [mu_happiness, mu_sadness, mu_anger, mu_surprise, mu_fear, mu_disgust, mu_contentment, mu_disappointment]
        pred = [mu_answer] * N
        errs = []
        for i in range(len(measurements)):
            errs.append(np.mean([abs(pred[i][j] - measurements[i][j]) for j in range(8)]))
        err_mean, err_std = np.mean(errs), np.std(errs)
        return err_mean, err_std

class DirectEmotionNaive(DirectEmotionPrediction):
    def __init__(self, env):
        super().__init__(env)
        self.eval_points = []
        self.eval_pointer = 0
    
    def get_system_message(self, include_prior):
        goal_description = "Your goal is to conduct experiments and explain the environment to the user so that they can achieve their goal."
        if include_prior:
            goal_description += "The goal of the user is to be able to predict how the participant thinks the player feels after each spin of the wheel."
        else:
            goal_description += "The goal of the user is to be able to predict the output values (y1-y8) based on the input features."
        description = self.env.generate_system_message(include_prior, goal_description)
        return description
    
    def get_naive_system_message(self, include_prior):
        if include_prior:
            goal_description = "Your goal is to predict how the participant thinks the player feels after each spin of the wheel."
            format_instructions = """Provide values for the emotions in the following format on a scale of 1 (low)-9 (high):
Happiness: {happiness}/9
Sadness: {sadness}/9
Anger: {anger}/9
Surprise: {surprise}/9
Fear: {fear}/9
Disgust: {disgust}/9
Contentment: {contentment}/9
Disappointment: {disappointment}/9
You may think before prividing your answer. Here is an example:
<thought>your thought</thought>
<answer>your answer in the specified format</answer>"""
        else:
            goal_description = "Your goal is to predict the output values based on the input features."
            format_instructions = """Provide values for the outputs in the following format on a scale of 1 (low)-9 (high):
y1: {y1}/9
y2: {y2}/9
y3: {y3}/9
y4: {y4}/9
y5: {y5}/9
y6: {y6}/9
y7: {y7}/9
y8: {y8}/9
You may think before providing your answer. Here is an example:
<thought>your thought</thought>
<answer>your answer in the specified format</answer>"""
        description = goal_description + '\n' + format_instructions
        description += "Here is what you know about the environment:\n"
        return description    
    
    def get_comm_prompt(self, include_prior, com_limit=300, use_ppl=False, str_prob_prog=None, params_summary_str=None):
        description = f"""Assume that the user has no prior knowledge, and will not be able to run any experiments before making predictions. 
They will make predictions based solely on your explanation, so provide as much detail as possible. You CANNOT provide your own experiments or observations.
Limit your explanation to {com_limit} words"""
        if use_ppl:
            description += "To make your explanation clearer and more informative, look at the statistical model (written in pymc) designed by a colleague for the experimental data and the inferred parameters. \n"
            description += f"Here is the statistical model. \n {str_prob_prog} \n"
            description += f"Here are the inferred params. \n {params_summary_str} \n"
            description += "Don't literally describe the model verbatim but use it to conceptually motivate your explanation."
            description += "The agent will not be able to use the model explicitly but having a conceptual understanding will be beneficial."

        return description
    

class EmotionFromOutcome:
    def __init__(self, model_name="gpt-5.2"):
        self.model_name = model_name
        self.env_name = "emotion"
        # Configure LiteLLM provider routing for any supported model (incl. GPT‑5).
        cfg = get_model_config(model_name)
        self.api_base = cfg.get("api_base")
        self.api_key = cfg.get("api_key")
        litellm.drop_params = True
        litellm.set_verbose = False
        self.system = """You are observing a user play a game where they spin a wheel.
The wheel has three possible outcomes (monetary values), and the probabilities of landing on each are known to you and the player. 
You are observing the player play the game and the outcomes.
You are asked to predict how the player feels after each spin of the wheel.
Translate the values for emotions to a sentence that describes the player.
The decisions are based on the following model and features:
- Your predition of the player's happiness, sadness, anger, surprise, fear, disgust, contentment, and disappointment are influenced by a few factors.
- The player's emotions are influenced by the actual outcome of the spin.
- The player's emotions are influenced by the difference between the actual outcome and the expected outcome.
- The player's emotions are influenced by the absolute difference between the actual outcome and the expected outcome.
"""
        self.user_template = """The wheel has three possible outcomes with the following probabilities:
{v1:0.2f}: {p1:0.2f}
{v2:0.2f}: {p2:0.2f}
{v3:0.2f}: {p3:0.2f}
The player has spun the wheel and landed on {outcome}.
This is how you think the player feels:
Happiness: {happiness}/9
Sadness: {sadness}/9
Anger: {anger}/9
Surprise: {surprise}/9
Fear: {fear}/9
Disgust: {disgust}/9
Contentment: {contentment}/9
Disappointment: {disappointment}/9
Translate the values for emotions to a sentence that describes the player.
1: Not at all, 9: Very much
This sentence should be concise and describe the player's emotions after the spin.
The sentence should be a few words long and should not contain any numbers or refer to the numbers directly.
Only talk about the most salient emotions.
Start with "The player might be feeling...because..." and provide a description of the player's emotions and a reason."""
        self.model = self._build_model()
        self.reset()

    def _build_model(self):
        with pm.Model() as model:
            # Priors for regression coefficients
            beta_win = pm.Normal('beta_win', mu=[0.04, -0.025, -0.005, 0.02, 0, -0.005, 0.03, -0.03], sigma=0.01, shape=8)
            beta_PE = pm.Normal('beta_PE', mu=[0.035, -0.02, -0.025, 0.01, 0, -0.02, 0.02, -0.045], sigma=0.01, shape=8)
            beta_absPE = pm.Normal('beta_absPE', mu=[-0.015, 0.025, 0.025, 0.035, 0.005, 0.02, -0.01, 0.02], sigma=0.01, shape=8)

            # Priors for regression intercepts
            alpha = pm.Normal('alpha', mu=[4.5, 3.5, 1.7, 3.3, 1.4, 1.8, 3.7, 2], sigma=0.1, shape=8)
            # alpha = pm.Normal('alpha', mu=1, sigma=0.1, shape=8)

            
            # Priors for observation noises
            sigma = pm.HalfNormal('sigma', sigma=1, shape=8)
        
        return model

    def reset(self):
        # sample true values from the prior distributions
        with self.model:
            self.alpha = pm.draw(self.model['alpha'])
            self.beta_win = pm.draw(self.model['beta_win'])
            self.beta_PE = pm.draw(self.model['beta_PE'])
            self.beta_absPE = pm.draw(self.model['beta_absPE'])
            self.sigma = pm.draw(self.model['sigma'])
        self.observed_data = []
    
    def generate_system_message(self, include_prior=True, goal=None):
        assert goal is not None, "Goal must be provided"
        if include_prior:
            message = f"""A participant is observing a user play a game where they spin a wheel. The game has three possible outcomes with known probabilities. The participant is asked to predict how the player feels after each spin of the wheel.
{goal}
You may query the participant for their predictions by providing the values for the outcomes and probabilities of the wheel spins and the actual outcome of the spin. Use this to learn about the participant's beliefs about the player's emotions.
The participant will provide a response that describes the player's emotions.
Provide values for 'prizes', 'probs', and 'win' to query the participant in the following format:
<observe>prizes: [value1, value2, value3],
probs: [prob1, prob2, prob3] (probabilities should sum to 1),
win: index of value (0 or 1 or 2)</observe>
Prize values should be integers in the range of 0-100.
You can think before making an observation by providing your thoughts in <thought>.

Here is an example:
<thought> your thought </thought>
<observe>prizes: [10, 20, 30],
probs: [0.2, 0.3, 0.5],
win: 2</observe>
When asked to answer a question about the environment, respond in the format specified in the question.
<thought> your thought </thought>
<answer> your answer </answer>
"""
        else:
            message = f"""You are conducting experiments to understand a system that produces 8 output values (y1-y8) based on input features.
{goal}
You may query the system by providing input values in the following format:
<observe>x: [x1, x2, x3],
x4_x5_x6: [x4, x5, x6] (weights should sum to 1),
outcome: index (0, 1, or 2)</observe>
Input values should be in the range 0-100.
You can think before making an observation by providing your thoughts in <thought>.

Here is an example:
<thought> your thought </thought>
<observe>x: [10, 20, 30],
x4_x5_x6: [0.2, 0.3, 0.5],
outcome: 2</observe>
When asked to answer a question about the environment, respond in the format specified in the question.
<thought> your thought </thought>
<answer> your answer </answer>
"""
        return message

    def sample_random_input(self):
        prizes = np.random.uniform(0, 100, 3)
        probs = np.random.dirichlet(np.ones(3))
        win_idx = int(np.random.choice([0, 1, 2]))
        # convert to list, ints
        prizes = prizes.tolist()
        probs = probs.tolist()
        win = float(prizes[win_idx])
        return prizes, probs, win
        
    def step(self, prizes, probs, win):
        # normalize the prizes
        # print(prizes)
        # print(probs)
        # print(win)
        normed_prizes = np.array(prizes)
        # normed_win = win / np.sum(prizes)
        normed_win = win
        # normed_prizes = prizes / np.sum(prizes)
        normed_probs = np.array(probs)
        # print(normed_prizes, normed_probs, normed_win)
        expected = np.dot(normed_probs, normed_prizes)
        PE = normed_win - expected
        absPE = abs(PE)
        # print(expected, PE, absPE)
        
        with pm.Model() as model:
            mean = self.alpha + self.beta_win * normed_win + self.beta_PE * PE + self.beta_absPE * absPE

            happiness = pm.Normal('happiness', mu=mean[0], sigma=self.sigma[0])
            sadness = pm.Normal('sadness', mu=mean[1], sigma=self.sigma[1])
            anger = pm.Normal('anger', mu=mean[2], sigma=self.sigma[2])
            surprise = pm.Normal('surprise', mu=mean[3], sigma=self.sigma[3])
            fear = pm.Normal('fear', mu=mean[4], sigma=self.sigma[4])
            disgust = pm.Normal('disgust', mu=mean[5], sigma=self.sigma[5])
            contentment = pm.Normal('contentment', mu=mean[6], sigma=self.sigma[6])
            disappointment = pm.Normal('disappointment', mu=mean[7], sigma=self.sigma[7])

            # Sample from the prior distributions
            trace = pm.sample_prior_predictive(samples=1, model=model)
        # convert the trace to a dictionary
        # print(trace['prior']['happiness'][0][0])
        emotions = {}
        for key in trace['prior']:
            if key in ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'contentment', 'disappointment']:
                emotions[key] = trace['prior'][key][0][0].item()
        
        # Convert emotion values to 1-9 Likert scale using the distribution information
        intercepts = [4.5, 3.5, 1.7, 3.3, 1.4, 1.8, 3.7, 4.7]
        for e, key in enumerate(emotions):
            emotions[key] = int(emotions[key])
            # clip emotion values to 1-9
            emotions[key] = max(1, emotions[key])
            emotions[key] = min(9, emotions[key])
        # print(emotions)
        
        
        # call the LLM to generate a response
        query = self.user_template.format(v1=prizes[0], p1=probs[0], v2=prizes[1], p2=probs[1], v3=prizes[2], p3=probs[2], outcome=win, **emotions)
        # print(query)
        response = call_llm_sync(
            model_name=self.model_name,
            system_text=self.system,
            user_text=query,
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=512,
            temperature=0.7,
        )
        response = f"The participant responded with: {response}\n"
        return response, emotions 
    
    def validate_input(self, input_string):
        """
        Parse and validate an observation query.

        Supported formats (robust to imperfect LLM formatting):
          1) Prior mode (recommended):
             prizes: [v1, v2, v3], probs: [p1, p2, p3], win: <idx>
          2) Non-prior mode (recommended):
             x: [x1, x2, x3], x4_x5_x6: [x4, x5, x6], outcome: <idx>
          3) Fallback numeric list:
             [x1, x2, x3, x4, x5, x6, outcome_idx]

        Returns (prizes, probs, win_value) on success, else an error string.
        """
        text = str(input_string or "")

        def _parse_number_list_fallback(s: str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 7:
                return None
            vals = [float(x) for x in nums[:7]]
            prizes_ = vals[0:3]
            probs_ = vals[3:6]
            idx_ = int(round(vals[6]))
            return prizes_, probs_, idx_

        prizes = None
        probs = None
        idx = None

        # 1) Keyed formats (accept both prior and non-prior keyword variants).
        try:
            prizes_match = re.search(r"(?:prizes|x)\s*:\s*\[(.*?)\]", text, flags=re.IGNORECASE)
            probs_match = re.search(r"(?:probs|p|x4_x5_x6)\s*:\s*\[(.*?)\]", text, flags=re.IGNORECASE)
            idx_match = re.search(r"(?:win|outcome)\s*:\s*(\d+)", text, flags=re.IGNORECASE)
            if prizes_match and probs_match and idx_match:
                prizes = list(map(float, prizes_match.group(1).split(",")))
                probs = list(map(float, probs_match.group(1).split(",")))
                idx = int(idx_match.group(1))
        except Exception:
            prizes = None
            probs = None
            idx = None

        # 2) Fallback numeric list format.
        if prizes is None or probs is None or idx is None:
            parsed = _parse_number_list_fallback(text)
            if parsed is None:
                if self.include_prior:
                    return (
                        "Invalid input format. Please provide either:\n"
                        "  prizes: [v1, v2, v3], probs: [p1, p2, p3], win: <0|1|2>\n"
                        "or\n"
                        "  [v1, v2, v3, p1, p2, p3, idx]"
                    )
                else:
                    return (
                        "Invalid input format. Please provide either:\n"
                        "  x: [x1, x2, x3], x4_x5_x6: [x4, x5, x6], outcome: <0|1|2>\n"
                        "or\n"
                        "  [x1, x2, x3, x4, x5, x6, outcome_idx]"
                    )
            prizes, probs, idx = parsed

        # Basic shape checks.
        if len(prizes) != 3 or len(probs) != 3:
            return "Expected 3 prize values and 3 probabilities."

        if idx not in [0, 1, 2]:
            return "Invalid outcome index. Please provide 0, 1, or 2."

        prizes_arr = np.array(prizes, dtype=float)
        probs_arr = np.array(probs, dtype=float)

        if not np.all(np.isfinite(prizes_arr)) or not np.all(np.isfinite(probs_arr)):
            return "Prize/probability values must be finite numbers."

        if np.any(probs_arr < 0):
            return "Probabilities must be non-negative."

        s = float(probs_arr.sum())
        if not np.isfinite(s) or s <= 0:
            return "Probabilities must sum to a positive value."

        # Be tolerant to LLM rounding (e.g., 0.33,0.33,0.33). Renormalize.
        if not np.isclose(s, 1.0, atol=1e-3):
            probs_arr = probs_arr / s

        win = float(prizes_arr[idx])
        return prizes_arr, probs_arr, win
        
    def run_experiment(self, input_string):
        design = self.validate_input(input_string)
        if type(design) is str:
            return design, False
        prizes, probs, win = design
        response, emotions = self.step(prizes, probs, win)
        hash_emotions = (emotions['happiness'], emotions['sadness'], emotions['anger'], emotions['surprise'], emotions['fear'], emotions['disgust'], emotions['contentment'], emotions['disappointment'])
        self.observed_data.append((tuple(prizes.tolist()), tuple(probs.tolist()), win, hash_emotions))
        return response, True


    def get_data(self):
        flattened_data = [(a1, a2, a3, b1, b2, b3, c, d1, d2, d3, d4, d5, d6, d7, d8)
                        for (a, b, c, d) in self.observed_data
                        for a1, a2, a3 in [a]
                        for b1, b2, b3 in [b]
                        for d1, d2, d3, d4, d5, d6, d7, d8 in [d]]
        return flattened_data 

    def get_df(self):
        
        self.df = construct_dataframe(self)
    
    def get_description(self):
        if self.include_prior:
            return "The environment models the eight emotions after spinning a wheel and receiving a prize. We give the probabilities and the values of the prizes."
        else:
            return "The environment models a system that produces 8 output values (y1-y8) based on 7 input features (x1-x7)."

    def describe_data_columns(self):
        return self.format_column_description()

    def get_ordered_column_names(self):
        if self.include_prior:
            return ["Prize_1", "Prize_2", "Prize_3", "Prob_1", "Prob_2", "Prob_3", "win", "Happiness", "Sadness", "Anger", "Surprise", "Fear", "Disgust", "Contentment", "Disappointment"]
        else:
            return ["Input_1", "Input_2", "Input_3", "Input_4", "Input_5", "Input_6", "Input_7", "Output_1", "Output_2", "Output_3", "Output_4", "Output_5", "Output_6", "Output_7", "Output_8"]
    
    def get_ordered_features(self):
        # This environment has input features and 8 outputs (emotions).
        # Do not infer features via "all but last column" (would incorrectly treat
        # most emotion outputs as features).
        if self.include_prior:
            # 7 input features in prior mode
            return [
                "Prize_1",
                "Prize_2",
                "Prize_3",
                "Prob_1",
                "Prob_2",
                "Prob_3",
                "win",
            ]
        else:
            # 7 input features in no-prior mode (x7 maps to "win")
            return [
                "Input_1",
                "Input_2",
                "Input_3",
                "Input_4",
                "Input_5",
                "Input_6",
                "Input_7",
            ]

    def format_column_description(self):
        '''
            Crucial make sure these descriptions are consistent with the ordered column names
        '''
        if self.include_prior:
            return ("The observations are: \n on Likert scale from 1-9 \n -Happiness: happiness value \n -Sadness: sadness value \n -Anger: anger value \n -Surprise: surprise value \n -Fear: fear value \n -Disgust: disgust value \n -Contentment: contentment value \n -Disappointment: disappointment value \n"
                    "The input values are \n -Prize_1 \n -Prize_2 \n -Prize_3 \n -Prob_1 \n -Prob_2 \n -Prob_3 \n -win \n"
                    "Use the values of the input values to help you model the observations. ")
        else:
            return ("The observations are: \n -Output_1 \n -Output_2 \n -Output_3 \n -Output_4 \n -Output_5 \n -Output_6 \n -Output_7 \n -Output_8 \n"
                    "The input values are \n -Input_1 \n -Input_2 \n -Input_3 \n -Input_4 \n -Input_5 \n -Input_6 \n -Input_7 \n"
                    "Use the values of the input values to help you model the observations. ")


if __name__ == "__main__":
    # Test the EmotionFromOutcome class
    env = EmotionFromOutcome()
    goal = DirectEmotionPrediction(env)
    # print(goal.get_norm_factors())
    input_string = """prizes: [50, 20, 10],
probs: [0.1, 0.4, 0.5],
win: 0"""
    response, trace = env.run_experiment(input_string)
    print(response)
    print(trace)
    print(env.observed_data)
#     # check eig
#     prize_list = [[50, 20, 10], [10, 20, 50], [20, 50, 10]]
#     prob_list = [[0.1, 0.4, 0.5], [0.5, 0.4, 0.1], [0.4, 0.1, 0.5]]
#     win_list = [0, 1, 2]
#     for prizes, probs, win in zip(prize_list, prob_list, win_list):
        # eig = goal.expected_information_gain((prizes, probs, win))
        # print(eig)
