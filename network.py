import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# 状态向量化函数，保持与之前一致
def state_to_vector(state_dict, num_nodes):
    """
    将字典状态转换为仅包含供应商信息的向量表示。

    参数:
        state_dict (dict): 输入的状态字典。
        num_nodes (int): 节点的总数量。

    返回:
        np.array: 仅包含供应商信息的状态向量。
    """
    state_vector = []

    for i in range(num_nodes):
        node_key = f"Node_{i}"
        node_data = state_dict.get(node_key, {'supplier': -1})  # 默认值为 -1，表示没有供应商信息

        # 只添加 supplier 信息
        state_vector.append(node_data['supplier'])

    return np.array(state_vector)

class ValueNet(nn.Module):
    """
    估值网络，用于评估当前状态的价值。

    参数:
        dim_state (int): 状态的维度。
    """
    def __init__(self, dim_state):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state):
        """
        前向传播计算状态的价值。

        参数:
            state (Tensor): 输入的状态。

        返回:
            Tensor: 状态的价值。
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, dim_state, num_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)  # 移除 softmax
        return logits  # 直接返回 logits

    def masked_policy(self, state, mask):
        logits = self.forward(state)  # 原始 logits
        # masked_logits = logits + (mask + 1e-45).log()
        masked_logits = logits + (mask + 1e-50).log()  # 使用 log(mask) 使无效动作的概率为 0
        return masked_logits

    def sample_action(self, state, mask):
        masked_logits = self.masked_policy(state, mask)
        masked_distribution = Categorical(logits=masked_logits)  # 正确：传递 logits
        action = masked_distribution.sample()
        log_prob = masked_distribution.log_prob(action)
        return action, log_prob


# 动态生成动作掩码的函数
def create_action_mask(state_dict, num_nodes, agent):
    """
    根据状态字典生成动作掩码，确保只为 `supplier` 为 -1 且其连接的节点的供应商与当前 agent 不同的节点选择供应商。

    在获取连接节点的供应商编号之前，确保连接的所有节点都已经选择了供应商。

    参数:
        state_dict (dict): 当前状态字典。
        num_nodes (int): 节点的总数量。
        agent (str): 当前的 agent，格式为 "agent_x"，其中 x 是供应商编号。

    返回:
        Tensor: 动作掩码。有效动作的位置为 1，无效动作的位置为 0。
    """
    # 获取传入的 agent 对应的供应商编号
    agent_supplier = int(agent.split('_')[1])

    # 初始化掩码为全 0
    mask = torch.zeros(num_nodes, dtype=torch.float32)

    # 遍历所有节点
    for i in range(num_nodes):
        node_key = f"Node_{i}"
        node_data = state_dict.get(node_key, {})

        # 仅处理尚未选择供应商的节点
        if node_data.get('supplier', -1) == -1:
            # 获取该节点连接的节点
            connected_nodes = node_data.get('connections', [])

            # 检查所有连接的节点是否都已经有供应商
            all_connected_have_supplier = all(
                state_dict.get(f"Node_{n}", {}).get('supplier', -1) != -1 for n in connected_nodes
            )

            # 如果所有连接的节点都已经选择了供应商
            if all_connected_have_supplier:
                # 获取连接节点的供应商编号
                connected_suppliers = set(
                    state_dict.get(f"Node_{n}", {}).get('supplier', -1) for n in connected_nodes
                )

                # 如果连接节点的供应商超过一个，则允许动作
                if len(connected_suppliers) != 1:
                    mask[i] = 1  # 允许该动作
                # 如果连接节点的供应商只有一个且与当前 agent 的供应商不同，则允许动作
                elif len(connected_suppliers) == 1 and next(iter(connected_suppliers)) != agent_supplier:
                    mask[i] = 1  # 允许该动作
                # 否则，不允许动作（mask[i] 保持为 0）
            else:
                # 如果并非所有连接的节点都已经选择了供应商，则允许动作
                mask[i] = 1

    return mask

class MAC(nn.Module):
    def __init__(self, num_agents=1, num_states=6, num_actions=5, gamma=0.95, tau=0.01):
        super().__init__()
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        # 为每个智能体初始化一个策略网络
        # self.agent2policy = {f"agent_{i}": PolicyNet(num_states, num_actions) for i in range(num_agents)}
        self.agent2policy = nn.ModuleDict({f"agent_{i}": PolicyNet(num_states, num_actions) for i in range(num_agents)})

        # 初始化价值网络
        self.value_net = ValueNet(num_states)
        self.target_value_net = ValueNet(num_states)
        self.target_value_net.load_state_dict(self.value_net.state_dict())  # 同步初始参数

    def policy(self, observation, agent, mask):
        """
        根据观察和指定的智能体生成动作和相应的对数概率。

        参数:
            observation (Tensor): 输入的观察。
            agent (str): 指定的智能体。
            mask (Tensor): 动作掩码，标识哪些动作是有效的。

        返回:
            tuple: 动作和对数概率。
        """
        action, log_prob = self.agent2policy[agent].sample_action(observation, mask)
        return action.item(), log_prob

    def value(self, observation):
        """
        计算给定观察的价值。
        """
        return self.value_net(observation)

    def target_value(self, observation):
        """
        计算给定观察的目标价值。
        """
        return self.target_value_net(observation)

    def compute_policy_loss(self, bs, br, bd, bns, logp_action_dict):
        """
        计算策略损失。
        """
        with torch.no_grad():
            td_value = self.target_value(bns).squeeze()
            td_value = br + self.gamma * td_value * (1 - bd)
            predicted_value = self.value(bs).squeeze()
            advantage = td_value - predicted_value
            # advantage = predicted_value - br

        policy_loss = 0
        for i in range(self.num_agents):
            policy_loss += -logp_action_dict[f"agent_{i}"] * advantage
        return policy_loss.mean()

    def compute_value_loss(self, bs, br, bd, bns, blogp_action_dict):
        """
        计算价值损失。
        """
        # td_value = br
        with torch.no_grad():
            td_value = self.target_value(bns).squeeze()
            td_value = br + self.gamma * td_value * (1 - bd)
        predicted_value = self.value(bs).squeeze()
        return F.mse_loss(predicted_value, td_value)

    def update_target_value(self):
        """
        更新目标价值网络的参数。
        """
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
