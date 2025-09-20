import gym

from gym.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime

start_time = datetime(2018, 1, 1, 0, 0, 0)
meal_scenario = CustomScenario(start_time=start_time, scenario=[(1, 20)])

register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={
        'patient_name': 'adolescent#002',
        'custom_scenario': meal_scenario
    }
)

env = gym.make('simglucose-adolescent2-v0')

observation = env.reset()
for t in range(100):
    env.render(mode='human')
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break
