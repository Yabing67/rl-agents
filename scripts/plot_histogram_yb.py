# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     plot_histogram
   Description :
   Author :       yabing
   date：          2021/8/3
-------------------------------------------------

"""

import matplotlib.pyplot as plt

def dqn_vs_ppo(name, dqn_list, ppo_list):
    head_list = ['1 head', '2 heads', '8 heads']
    x = list(range(len(head_list)))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, dqn_list, width=width, label='dqn', fc='blue')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, ppo_list, width=width, label='ppo', tick_label=head_list, fc='orange')
    plt.title(f'{name}')
    plt.legend()
    plt.savefig('./'f'{name}''.png')
    plt.show()

def compare_collision():
    name = 'collisions of dqn and ppo with different heads'
    collisions_dqn = [0, 2, 0]
    collisions_ppo = [11, 35, 21]
    dqn_vs_ppo(name, collisions_dqn, collisions_ppo)

def compare_episode():
    name = 'episode length of dqn and ppo with different heads'
    episode_dqn = [1.168129098, 2.244119575, 2.455248001]
    episode_ppo = [1.655892642, 1.44601141, 1.932915423]
    dqn_vs_ppo(name, episode_dqn, episode_ppo)

if __name__ == "__main__":
    compare_collision()
    compare_episode()
