#!/usr/bin/python
# -*- coding: utf-8 -*-
from Agent import AgentDiscretePPO
from core import ReplayBuffer
from draw import Painter
from env4Snake import Snake
import random
import pygame
import numpy as np
import torch
import matplotlib.pyplot as plt

def testAgent(test_env, agent, episode):
    ep_reward = 0
    o = test_env.reset()
    for _ in range(650):
        if episode % 100 == 0:
            test_env.render()
        for event in pygame.event.get():  # 不加这句render要卡，不清楚原因
            pass
        a_int, a_prob = agent.select_action(o)
        o2, reward, done, _ = test_env.step(a_int)
        ep_reward += reward
        if done:
            break
        o = o2
    return ep_reward

if __name__ == "__main__":
    env = Snake()
    test_env = Snake()
    act_dim = 4
    obs_dim = 6
    agent = AgentDiscretePPO()
    agent.init(512, obs_dim, act_dim, if_use_gae=True)
    agent.state = env.reset()
    buffer = ReplayBuffer(2**12, obs_dim, act_dim, True)
    MAX_EPISODE = 200
    batch_size = 64
    rewardList = []
    maxReward = -np.inf

    episodeList = []  # 存储训练轮数
    rewardArray = []  # 存储rewards得分

    for episode in range(MAX_EPISODE):
        with torch.no_grad():
            trajectory_list = agent.explore_env(env, 2**12, 1, 0.99)
        buffer.extend_buffer_from_list(trajectory_list)
        agent.update_net(buffer, batch_size, 1, 2**-8)
        ep_reward = testAgent(test_env, agent, episode)
        print('Episode:', episode, 'Reward:%f' % ep_reward)
        rewardList.append(ep_reward)

        episodeList.append(episode)
        rewardArray.append(ep_reward)

        if episode > MAX_EPISODE / 3 and ep_reward > maxReward:
            maxReward = ep_reward
            print('保存模型！')
            torch.save(agent.act.state_dict(), 'model_weights/act_weight.pkl')
    pygame.quit()

    painter = Painter(load_csv=True, load_dir='reward.csv')
    painter.addData(rewardList, 'PPO')
    painter.saveData('reward.csv')
    painter.setTitle('snake game reward')
    painter.setXlabel('episode')
    painter.setYlabel('reward')
    painter.drawFigure()

    # 进行曲线拟合
    poly_degree = 3  # 多项式的次数
    coeffs = np.polyfit(episodeList, rewardArray, poly_degree)
    poly = np.poly1d(coeffs)
    fitted_rewards = poly(episodeList)

    # 绘制训练轮数与rewards得分曲线
    plt.plot(episodeList, rewardArray, label='Actual Rewards')
    plt.plot(episodeList, fitted_rewards, label='Fitted Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # 保存曲线图片
    plt.savefig('reward_curve.png')

    # 显示绘制的图形
    plt.show()
