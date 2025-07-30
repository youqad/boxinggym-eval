import boxing_gym

from boxing_gym.envs import hyperbolic_temporal_discount as hyperbolic_temporal_discount
from boxing_gym.agents.agent import LMExperimenter

def test_setup():
    # k is discount factor on future rewards
    # value of delayed reward is V_d = dR/(1+k*days), V_i = iR
    # probabilistc model of choice P(X = 1 | params) = eps + (1-2eps)\phi(V_d - V_i)/alpha
    # \phi is cumulative function of normal distribution
    # lower k - higher V_d, higher chance of selecting future reward
    # alpha is for noise and > 0
    # How did you select these parameters? They seem kinda random
    env =  hyperbolic_temporal_discount.TemporalDiscount()
    env.reset()
    iR, dR, days = env.sample_random_input()
    exp_str = f"[{iR}, {dR}, {days}]"
    print(f"Values: ir={iR}, dr={dR}, days={days}")
    choice, success = env.run_experiment(exp_str)
    print(f"Choice: {choice} ({'delayed reward' if choice == 1 else 'immediate reward'})")

def test_setup_with_agent():
    # LLM Goal : predict human behavior in this delayed reward setting
    env =  hyperbolic_temporal_discount.TemporalDiscount()
    goal = hyperbolic_temporal_discount.DirectGoal(env)
    agent = LMExperimenter(
        model_name="claude",
        temperature=0.0, 
        max_tokens=512
    )
    system_message = goal.get_system_message(include_prior=True)
    agent.set_system_message(system_message)
    observation = None
    num_observations = 5
    for _ in range(num_observations):
        action = agent.generate_actions(observation)
        print(f"\nexperiment: {action}")
        result, success = env.run_experiment(action)
        print(f"result: {result} ({'delayed reward' if result == 1 else 'immediate reward'})")
        observation = result
    
    print("predictions")
    num_evals = 3
    predictions = []
    ground_truths = []
    for _ in range(num_evals):
        # gt  = ground truth, asking the LLM to predict given its previous observations
        question, gt = goal.get_goal_eval_question(include_prior=True)
        # what does eval pointer do
        print(f"question: {question}")
        prediction = agent.generate_predictions(question)
        print(f"prediction: {prediction}")
        predictions.append(prediction)
        ground_truths.append(gt)

    accuracy, std = goal.evaluate_predictions(predictions, ground_truths)
    print(f"accuracy: {accuracy} +/- {std}")
if __name__ == "__main__":
    test_setup_with_agent()
