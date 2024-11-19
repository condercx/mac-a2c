import torch
import numpy as np
from collections import defaultdict

class Rollout:
    """
    用于存储和处理环境与智能体交互的数据。
    """

    def __init__(self):
        self.state_list = []
        self.reward_list = []
        self.done_list = []
        self.next_state_list = []
        self.logp_actions_dict = defaultdict(list)

    def put(self, state, reward, done, next_state, logp_action_dict):
        """
        将一次交互数据放入Rollout实例中。
        """
        # print(f"Putting state type: {type(state)}")
        # print(f"Putting next_state type: {type(next_state)}")
        self.state_list.append(state)  # state 应为 NumPy 数组
        self.reward_list.append(reward)
        self.done_list.append(done)
        self.next_state_list.append(next_state)  # next_state 应为 NumPy 数组
        for k, v in logp_action_dict.items():
            self.logp_actions_dict[k].append(v)  # v 为张量，位于 GPU

    def tensor(self):
        """
        将存储的数据转换为张量。
        """
        # print("Converting state_list to tensor...")
        # for i, s in enumerate(self.state_list[:5]):
            # print(f"state_list[{i}] type: {type(s)}")
        
        # 确保 state_list 中的所有元素都是 NumPy 数组
        states = []
        for s in self.state_list:
            if isinstance(s, np.ndarray):
                states.append(s)
            elif isinstance(s, torch.Tensor):
                states.append(s.cpu().numpy())
            else:
                raise ValueError("state_list contains unsupported type.")

        bs = torch.tensor(np.asarray(states)).float()

        # print("Converting next_state_list to tensor...")
        next_states = []
        for ns in self.next_state_list:
            if isinstance(ns, np.ndarray):
                next_states.append(ns)
            elif isinstance(ns, torch.Tensor):
                next_states.append(ns.cpu().numpy())
            else:
                raise ValueError("next_state_list contains unsupported type.")

        bns = torch.tensor(np.asarray(next_states)).float()
        br = torch.tensor(np.asarray(self.reward_list)).float()
        bd = torch.tensor(np.asarray(self.done_list)).float()

        # 处理 logp_action_dict，将所有 log_probs 移动到 CPU
        blogp_action_dict = {}
        for k, v in self.logp_actions_dict.items():
            blogp_action_dict[k] = torch.stack(v).cpu()

        return bs, br, bd, bns, blogp_action_dict
