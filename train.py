import numpy as np
import gym

env = gym.make('CartPole-v0')
env.reset()


def get_noise():
    return np.random.rand(4) * 2 - 1


def take_action(parameters):
    observation = env.reset()
    step_reward = 0
    for _ in range(200):
        env.render()
        if np.matmul(observation, parameters) < 0:
            action = 0
        else:
            action = 1

        observation, reward, done, info = env.step(action)

        step_reward += reward

        if done:
            break
    return step_reward


def train():
    noise_scale = 0.4
    max_reward = 0

    parameters = get_noise()

    while True:
        newparams = parameters + get_noise() * noise_scale
        reward = take_action(parameters)
        if reward > max_reward:
            max_reward = reward
            parameters = newparams

        print('max reward is ', max_reward, '. reward is ', reward)
        if reward == 200:
            break

    test(parameters)


def test(parameters):
    print ' train over. test begin.'
    step_reward = 0
    for _ in range(10):
        step_reward += take_action(parameters)

    print('Final reward is ', step_reward / 10)

if __name__=="__main__":
    train()
