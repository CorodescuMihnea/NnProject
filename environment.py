import gym
import datetime
import os
import numpy as np

from agent import DeepQAgent


def main():
    env = gym.make("LunarLander-v2")

    timestamp = '{:%Y-%m-%d-%H:%M}'.format(datetime.datetime.now())
    o_dir = "LunarLander-v2/{}/models".format(timestamp)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    nof_episodes = 500
    # 8 values in [0, 1]
    state_size = env.observation_space.shape[0]
    # 0, 1, 2, 3
    action_size = env.action_space.n
    agent = DeepQAgent(state_size, action_size, model=2)
    batch_size = 32

    for episode in range(nof_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        t = 0
        episode_reward = 0
        # Iterate over the timesteps
        while not done:
            env.render()

            # Instruct the agent to choose an action based on the current state of the environment
            # This may be a random action depending on the value of the exploration_rate(epsilon)
            action = agent.act(state)
            # Execute said action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            next_state = np.reshape(next_state, [1, state_size])

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, time: {}, total_reward: {}"
                      .format(episode, nof_episodes - 1, t, episode_reward))
            t += 1
        if len(agent.memory) / batch_size > 1:
            agent.train(batch_size)
        # Save model after training
        if episode % batch_size == 1:
            agent.save(o_dir + "/model_" + str(episode) + ".hdf5")


if __name__ == "__main__":
    main()
