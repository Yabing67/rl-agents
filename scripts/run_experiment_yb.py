# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_experiment_yb
   Description :
   Author :       yabing
   date：          2021/7/20
-------------------------------------------------

"""
import sys
sys.path.insert(0, '/home/yabing/KIT/MRT_masterThesis/MRT_MA/highway-env/scripts/')
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment
# from utils import show_videos
import time

# Get the environment and agent configurations from the rl-agents repository
env_config = 'configs/ExitEnv/env.json'
# agent_config = 'configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json'
# agent_config = 'configs/IntersectionEnv/agents/DQNAgent/ego_attention_8h.json'
# agent_config = 'configs/IntersectionEnv/agents/DQNAgent/ego_attention.json'
# agent_config = 'configs/IntersectionEnv/agents/DQNAgent/baseline.json'
agent_config = 'configs/IntersectionEnv/agents/DQNAgent/self_attention_2h.json'
env = load_environment(env_config)
env.configure({"offscreen_rendering": False})
agent = load_agent(agent_config, env)
num_episodes = 100
# model = 'out/ExitEnv/DQNAgent/saved_models/exit_egoattention_1000episode.tar'  # 2 head
# model = 'out/ExitEnv/DQNAgent/run_20210722-181320_4036/checkpoint-final.tar'  # 2 head
# model = 'out/ExitEnv/DQNAgent/run_20210731-115204_7029/checkpoint-final.tar'  # 8 head
# model = 'out/ExitEnv/DQNAgent/run_20210731-120550_10093/checkpoint-final.tar'  # 1 head
# model = 'out/ExitEnv/DQNAgent/run_20210731-180658_18391/checkpoint-final.tar'  # baseline
model = 'out/ExitEnv/DQNAgent/run_20210731-223905_23660/checkpoint-final.tar'  # self attention 2 head
model = 'out/ExitEnv/DQNAgent/run_20210731-214351_15056/checkpoint-final.tar'  # self attention 1 head
show_env = True

def main():
    train = False
    if train:
        print(f"Ready to train {agent} on {env}")
        evaluation = Evaluation(env, agent, num_episodes=num_episodes, display_env=False)
        evaluation.train()
    else:
        print(f"Ready to test {agent} on {env}")
        env.configure({"offscreen_rendering": False})
        evaluation = Evaluation(env, agent, num_episodes=num_episodes, recover=model, display_env=False, display_agent=show_env)
        time_start = time.time()
        collisions = 0
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                evaluation.agent.eval()  # add this sentence so can make sure that the agent is loaded from checkpoint
                actions = evaluation.agent.plan(obs)
                action = actions[0]
                obs, reward, done, info = evaluation.env.step(action)
                # obs, reward, done, info = evaluation.monitor.step(action)
                env.render()
                if info["crashed"]:
                    collisions += 1
        time_end = time.time()
        total_time = time_end - time_start
        print('total time cost', total_time, 's')
        print('time per episode', total_time / num_episodes, 's')
        print('total number of collisions：', collisions)
        # show_videos(evaluation.run_directory)


if __name__ == "__main__":
    main()