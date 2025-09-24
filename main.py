from datetime import datetime
import gym
from gym.envs.registration import register
import numpy as np
from ppo import Agent
from utils import plot_learning_curve
from simglucose.simulation.scenario_gen import RandomScenario
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()  # Tensorboard


start_time = datetime(2018, 1, 1, 0, 0, 0)
meal_scenario = RandomScenario(start_time=start_time, seed=1)

# Testing every patients
'''
patient_names = [
    'adolescent#001','adolescent#002','adolescent#003','adolescent#004','adolescent#005',
    'adolescent#006','adolescent#007','adolescent#008','adolescent#009','adolescent#010',
    'adult#001','adult#002','adult#003','adult#004','adult#005',
    'adult#006','adult#007','adult#008','adult#009','adult#010',
    'child#001','child#002','child#003','child#004','child#005',
    'child#006','child#007','child#008','child#009','child#010',
]
'''
# Testing one patients
patient_names = 'adult#001'

register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={
        'patient_name': patient_names,
        'custom_scenario': meal_scenario
    }
)


if __name__ == '__main__':

    env = gym.make('simglucose-adolescent2-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = Agent(
        n_actions=1,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape
    )
    n_games = 300

    figure_file = 'plots/cartpole.png'
    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            'episode', i,
            'score %.1f' % score,
            'avg score %.1f' % avg_score,
            'time_steps', n_steps,
            'learning_steps', learn_iters
        )

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
