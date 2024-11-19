import torch
import numpy as np
from torch.optim import Adam
from collections import defaultdict
import os
import matplotlib
import matplotlib.pyplot as plt
from rollout import Rollout
print(plt.get_backend())
matplotlib.use('Agg')
from network import state_to_vector, create_action_mask
from data_process import draw_network
import pandas as pd

def train(args, env, central_controller, topology_matrix, node_relationships, device):
    """
    训练多智能体系统。
    """
    # **将中央控制器的所有子模块移动到设备**
    central_controller.to(device)

    # 初始化策略网络参数列表
    policy_params = []
    for i in range(args.num_agents):
        policy_params += list(central_controller.agent2policy[f"agent_{i}"].parameters())
    policy_optimizer = Adam(policy_params, lr=args.lr_policy)
    value_optimizer = Adam(central_controller.value_net.parameters(), lr=args.lr_value)

    # 初始化回合奖励列表和日志记录
    episode_reward_lst = []
    olg_lst = []
    value_loss_lst = []
    log = defaultdict(list)

    # 开始指定回合数的训练
    for episode in range(args.num_episode):
        # 在每个回合开始时重置环境状态和智能体观察
        env.reset()
        state = state_to_vector(env.observe("agent_0"), args.num_nodes)
        state = torch.as_tensor(state).float().to(device)
        logp_action_dict = {}
        episode_reward = 0
        done = False
        rollout = Rollout()

        while not done:
            # 遍历所有智能体，选择动作并逐步执行环境
            for agent in env.agent_iter():
                # 根据当前状态生成动作掩码
                mask = create_action_mask(env.observe("agent_0"), args.num_nodes, agent)
                # **将掩码移动到设备**
                mask = mask.to(device)
                if mask.sum() == 0:
                    env.next_agent()
                else:
                    # 使用策略网络根据状态和掩码选择动作
                    action, logp_action = central_controller.policy(state, agent, mask)
                    logp_action_dict[agent] = logp_action
                    env.step(action)
                # draw_network(topology_matrix, node_relationships)
                # 更新每一步的状态、奖励和结束标志
                if env.agent_selection == "agent_0":
                    next_state = state_to_vector(env.observe("agent_0"), args.num_nodes)
                    next_state = torch.as_tensor(next_state).float().to(device)
                    reward = env.rewards["agent_0"] + env.calculate_reward(state.cpu().numpy(), next_state.cpu().numpy(), node_relationships)
                    # **将状态和 next_state 作为 NumPy 数组传递**
                    rollout.put(state.cpu().numpy(), reward, done, next_state.cpu().numpy(), logp_action_dict)
                    state = next_state

                    episode_reward += reward
                    # draw_network(topology_matrix, node_relationships)
                    # 回合结束时更新模型并记录数据
                    if not env.terminations["agent_0"] and env.check_truncations():
                        env.truncations["agent_0"] = True
                        unassigned_nodes = sum(1 for node_data in env.nodes.values() if node_data['supplier'] == -1)
                        episode_reward -= unassigned_nodes * 100
                    terminations = env.terminations["agent_0"]
                    truncations = env.truncations["agent_0"]
                    done = terminations or truncations
                    if done:
                        olg_lst.append(100 - env.handle_trojan_nodes())
                        episode_reward_lst.append(episode_reward)
                        # 更新 node_relationships 中的 supplier 信息
                        for i in range(args.num_nodes):
                            node_key = f"Node_{i}"
                            node_relationships[node_key]['supplier'] = state[i].item()  # 更新 supplier
                        # 每20个回合保存模型并绘制奖励曲线
                        if episode % 20 == 0:
                            torch.save(
                                {agent: central_controller.agent2policy[agent].state_dict() for agent in central_controller.agent2policy},
                                os.path.join(args.output_dir, "model.pt"),
                            )
                            x_axis = np.arange(len(olg_lst))
                            plt.plot(x_axis, olg_lst)
                            plt.xlabel("Episode", fontsize=22)
                            plt.tick_params(axis='both', labelsize=18)
                            plt.xticks(np.linspace(0, 6000, 7))
                            plt.axis([0, 6000, 74, 100])
                            plt.savefig("Training Progress.pdf", bbox_inches="tight")
                            plt.close()
                            # 保存数据到CSV文件
                            df = pd.DataFrame({'Episode': np.arange(len(olg_lst)), 'olg_lst': olg_lst})
                            df.to_csv('olg_lst.csv', index=False)

                        # 将存储的数据转换为张量，计算并更新价值损失和策略损失
                        bs, br, bd, bns, blogp_action_dict = rollout.tensor()
                        # **将张量移动到设备**
                        bs = bs.to(device)
                        br = br.to(device)
                        bd = bd.to(device)
                        bns = bns.to(device)
                        for k in blogp_action_dict:
                            blogp_action_dict[k] = blogp_action_dict[k].to(device)

                        value_loss = central_controller.compute_value_loss(bs, br, bd, bns, blogp_action_dict)
                        value_optimizer.zero_grad()
                        value_loss.backward()
                        value_optimizer.step()

                        policy_loss = central_controller.compute_policy_loss(bs, br, bd, bns, blogp_action_dict)
                        policy_optimizer.zero_grad()
                        policy_loss.backward()
                        policy_optimizer.step()

                        central_controller.update_target_value()

                        value_loss_lst.append(value_loss.item())
                        # 每20个回合保存模型并绘制loss曲线
                        if episode % 20 == 0:
                            x_axis = np.arange(len(value_loss_lst))
                            plt.plot(x_axis, value_loss_lst)
                            plt.xlabel("Episode", fontsize=22)
                            plt.ylabel("Value Loss", fontsize=22)
                            plt.tick_params(axis='both', labelsize=18)
                            plt.xticks(np.linspace(0, 6000, 7))
                            plt.axis([0, 6000, 0, 600])
                            plt.savefig("Value Loss over Episodes.pdf", bbox_inches="tight")
                            plt.close()
                            # 保存数据到CSV文件
                            df = pd.DataFrame({'Episode': np.arange(len(value_loss_lst)), 'Value Loss': value_loss_lst})
                            df.to_csv('value_loss_data.csv', index=False)

                        # 记录损失数据，每20个回合打印训练信息
                        log["value_loss"].append(value_loss.item())
                        log["policy_loss"].append(policy_loss.item())
                        if episode % 20 == 0:
                            avg_value_loss = np.mean(log["value_loss"][-20:])
                            avg_policy_loss = np.mean(log["policy_loss"][-20:])
                            avg_reward = np.mean(episode_reward_lst[-20:])
                            print(
                                f"episode={episode}, moving reward={avg_reward:.2f}, value loss={avg_value_loss:.4f}, policy loss={avg_policy_loss:.4f}")

                        break
