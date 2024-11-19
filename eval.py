import torch
import numpy as np
import time
from network import MAC
from env import SimpleEnv
import os


def eval(args):
    """
    评估多智能体协作模型在简单传播环境中的性能。

    参数:
    - args: 包含仿真参数的命名空间，如代理数量、状态数、操作数、输出目录等。
    """
    # 初始化环境，设置环境参数
    env = SimpleEnv(num_agents=args.num_agents, max_cycles=25, state_dim=args.num_states)
    # 初始化多智能体控制器
    central_controller = MAC(num_agents=args.num_agents, num_states=args.num_states, num_actions=args.num_actions)

    # 加载训练好的策略网络
    agent2policynet = torch.load(os.path.join(args.output_dir, "model.pt"))
    # 将保存的状态字典加载到控制器的策略网络中
    for agent, state_dict in agent2policynet.items():
        central_controller.agent2policy[agent].load_state_dict(state_dict)

    # 将控制器设置为评估模式
    central_controller.eval()

    # 用于记录每轮的总奖励
    episode_reward_lst = []
    for episode in range(10):
        # 每轮开始前，总奖励归零
        episode_reward = 0

        # 重置环境，开始新的回合
        env.reset()
        # 遍历每个代理的回合
        for agent in env.agent_iter():
            # 观察当前所有代理的状态
            state = [env.observe(f"agent_{x}") for x in range(args.num_agents)]
            state = np.concatenate(state)

            # 根据当前状态选择动作
            action, _ = central_controller.policy(torch.as_tensor(state).float(), agent)
            # 执行动作，推进环境一步
            env.step(action)

            # 检查是否是第一个代理的回合结束，用于计算回合奖励和判断回合是否结束
            if env.agent_selection == "agent_0":
                # 更新状态
                next_state = [env.observe(f"agent_{x}") for x in range(args.num_agents)]
                next_state = np.concatenate(next_state)
                # 获取当前回合的奖励
                reward = env.rewards["agent_0"]
                # 判断当前回合是否结束
                done = env.terminations["agent_0"] or env.truncations["agent_0"]
                state = next_state

                # 累加奖励
                episode_reward += reward

                # 为了观察，暂停0.1秒
                time.sleep(0.1)

                # 如果回合结束，记录该回合的总奖励，并计算最近20个回合的平均奖励
                if done:
                    episode_reward_lst.append(episode_reward)
                    avg_reward = np.mean(episode_reward_lst[-20:])
                    # 打印回合信息和平均奖励
                    print(f"episode={episode}, episode reward={episode_reward}, moving reward={avg_reward:.2f}")
                    break