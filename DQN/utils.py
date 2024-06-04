# https://github.com/Murraaa11/Algorithmic-Trading-with-DQN/blob/main/DQN_Trading/main.py

import torch
import datetime as dt

def train(cfg, env, agent):
    ''' training
    '''
    print('Start Training!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    rewards = []  # record total rewards
    ma_rewards = []  # record moving average total rewards
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            # print(f"Action: " + str(action)), action = 0 or 1 or 2
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # update target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('Episode：{}/{}, Reward：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('Finish Training!')
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('Start Testing!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    ############# Test does not use e-greedy policy, so we set epsilon to 0 ###############
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    ################################################################################
    stocks = env.tickers
    rewards = []  # record total rewards
    for i_ep in range(len(stocks)):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        print(f"Episode：{i_ep + 1}/{len(stocks)}，Reward：{ep_reward:.1f}")
    print('Finish Testing!')
    return stocks, rewards


class Config:
    def __init__(self):
        ################################## env hyperparameters ###################################
        self.algo_name = 'DQN' # algorithmic name
        self.env_name = 'TradingSystem' # environment name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # examine GPU
        self.seed = 11 # random seed
        self.train_eps = 200 # training episodes
        self.state_space_dim = 50 # state space size (K-value)
        self.action_space_dim = 3 # action space size (short: 0, neutral: 1, long: 2)
        ################################################################################

        ################################## algo hyperparameters ###################################
        self.gamma = 0.95  # discount factor
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy
        self.epsilon_decay = 500  # attenuation rate of epsilon in e-greedy policy
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 1000  # capacity of experience replay
        self.batch_size = 64  # size of mini-batch SGD
        self.target_update = 4  # update frequency of target network
        self.hidden_dim = 128  # dimension of hidden layer
        ################################################################################

        ################################# save path ##############################
        curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.result_path = "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'
        self.model_path = "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'
        self.save = True  # whether to save the image
        ################################################################################