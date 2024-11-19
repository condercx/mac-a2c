from network import create_action_mask
import copy
import torch
class SimpleEnv:
    def __init__(self, node_data, num_agents=2):
        self.num_agents = num_agents
        self.nodes = node_data  # 节点状态字典
        self.current_cycle = 0
        self.agent_selection = None
        self.reset()

    def reset(self):
        # 初始化节点的供应商信息，设置为-1代表未选择供应商
        for node_key in self.nodes:
            self.nodes[node_key]['supplier'] = -1

        self.rewards = {f"agent_{i}": 0 for i in range(self.num_agents)}
        self.terminations = {f"agent_{i}": False for i in range(self.num_agents)}  # 终止标志
        self.truncations = {f"agent_{i}": False for i in range(self.num_agents)}  # 截断标志
        self.agent_selection = "agent_0"  # 初始选择 agent_0 开始
        self.current_cycle = 0
        return self._get_obs()

    def _get_obs(self):
        # 返回每个节点的状态，包含供应商和连接信息
        return self.nodes

    def observe(self, agent_id):
        # 直接返回整个状态
        return self.nodes

    def step(self, action):
        """
        每个step，当前选择的智能体选择一个节点作为供应商，并更新节点的状态。
        Action 是选择的节点的 index。
        """
        agent_index = int(self.agent_selection.split('_')[-1])
        selected_node = f"Node_{action}"

        # 更新供应商为当前 agent
        self.nodes[selected_node]['supplier'] = agent_index

        # # 计算奖励
        # reward = self._calculate_reward(selected_node)

        # 检查所有节点是否都已经选择了供应商
        if self._all_nodes_assigned():
            # 遍历所有节点进行木马检测和删除操作
            # average_percentage = self._handle_trojan_nodes()
            # print(average_percentage)
            # 计算剩余节点的百分比并加到奖励中
            # remaining_percentage = self._calculate_remaining_percentage()
            # print(average_percentage)
            # reward += 0.01 * (100 - average_percentage) * len(self.nodes) * 0.1  # 加到当前奖励中
            # print(reward)
            # self.rewards["agent_0"] -= average_percentage * len(self.nodes) * 0.01
            self.terminations["agent_0"] = True

        # # 更新当前智能体的奖励
        # if self.agent_selection != "agent_0":
        #     self.rewards["agent_0"] += reward
        # else:
        #     self.rewards["agent_0"] = reward


        # 更新agent_selection
        next_agent_index = (agent_index + 1) % self.num_agents
        self.agent_selection = f"agent_{next_agent_index}"

        # 检查是否有节点连接的所有节点属于同一供应商
        # if self.check_truncations():
        #     self.truncations[self.agent_selection] = True
        #     self.rewards["agent_0"] -= self.num_agents  # 给予一个惩罚
        # for i in range(self.num_agents):
        # # 检查键是否存在
        #     if f"agent_{i}" in self.truncations and self.truncations[f"agent_{i}"]:
        #         self.rewards[self.agent_selection] -= 2
        return self._get_obs(), {}

    # def calculate_reward(self, selected_node):
    #     # """
    #     # 根据选择的节点，计算与其他节点连接后的奖励。
    #     # 奖励计算公式：连接不同供应商的数目 - 相同供应商的数目，再除以总连接数。
    #     # """
    #     # supplier = self.nodes[selected_node]['supplier']
    #     # connections = self.nodes[selected_node]['connections']
    #     #
    #     # different_supplier_count = 0
    #     # same_supplier_count = 0
    #     #
    #     # for conn_node in connections:
    #     #     conn_supplier = self.nodes[f"Node_{conn_node}"]['supplier']
    #     #     if conn_supplier == supplier:
    #     #         same_supplier_count += 1
    #     #     elif conn_supplier != -1:
    #     #         different_supplier_count += 1
    #     #
    #     # total_connections = len(connections)
    #     # if total_connections == 0:
    #     #     return 0
    #     #
    #     # # reward = (different_supplier_count - same_supplier_count) / total_connections
    #     # reward = -0.01 * (same_supplier_count / total_connections)
    #     # return reward

    def calculate_reward(self, state, next_state, node_relationships):
        """
        计算奖励基于每个节点从state到next_state的供应商变化后，其相连的相同供应商的节点集合的大小变化。
        首先识别在next_state中哪些节点的供应商发生了变化，然后通过广度优先搜索（BFS）计算与这些节点相连且供应商相同的节点集合的大小。
        如果这个集合的大小（包括该节点自身）大于2，奖励减1。

        :param state: 当前状态（每个节点的供应商，-1 表示未分配）
        :param next_state: 下一状态（每个节点的供应商，-1 表示未分配）
        :param node_relationships: 每个节点的连接信息
        :return: reward
        """
        reward = 0
        # 找出供应商发生变化的节点
        changed_nodes = [i for i, (s1, s2) in enumerate(zip(state, next_state)) if s1 != s2]
        # print(changed_nodes)
        # 计算奖励
        for node_index in changed_nodes:
            node_key = f"Node_{node_index}"
            current_supplier = next_state[node_index]  # 获取该节点在next_state中的供应商
            connections = node_relationships[node_key]['connections']

            # 通过广度优先搜索找出所有相连的且供应商相同的节点
            cluster = set()
            queue = [node_index]
            while queue:
                current_node_index = queue.pop(0)
                current_node_key = f"Node_{current_node_index}"
                if current_node_key not in cluster:
                    cluster.add(current_node_key)
                    # 获取当前节点的所有连接节点
                    connected_nodes = node_relationships[current_node_key]['connections']
                    # 对于每个连接节点，如果供应商相同，则添加到队列中
                    for neighbor_index in connected_nodes:
                        if next_state[neighbor_index] == current_supplier:
                            queue.append(neighbor_index)

            cluster_size = len(cluster)  # 包括节点自身
            # print(f"Cluster size for node {node_key}: {cluster_size}")
            # 如果计算出的簇大小大于2，则奖励减1
            # if cluster_size > 2:
            #     reward -= 1
            reward -= cluster_size
        return reward

    def check_truncations(self):
        """
        检查节点的供应商和它所连接的所有节点供应商是否相同，且所有节点的供应商已选择。
        如果某节点的供应商与其连接节点的供应商相同且都已选择供应商（不等于 -1），则终止。
        """
        # 初始化一个全零的掩码并集，大小与节点数量相同
        combined_mask = torch.zeros(len(self.nodes), dtype=torch.int)
        # 生成 agent 名称
        agent_names = [f"agent_{i}" for i in range(self.num_agents)]

        # 对于每个 agent，创建其动作掩码并与 combined_mask 进行并集操作
        for agent in agent_names:
            mask = create_action_mask(self.nodes, len(self.nodes), agent)
            mask = mask.to(dtype=torch.int)
            combined_mask = combined_mask | mask  # 使用按位或操作
        if combined_mask.sum() == 0:
            return True
        # for node_key, node_data in self.nodes.items():
        #     supplier = node_data['supplier']
        #     if supplier == -1:
        #         continue  # 如果节点未选择供应商，则跳过
        #
        #     # 获取所有连接节点的供应商
        #     connected_suppliers = set(self.nodes[f"Node_{n}"]['supplier'] for n in node_data['connections'])
        #
        #     # 检查该节点和所有连接节点的供应商是否相同且都已选择供应商
        #     if len(connected_suppliers) == 1 and supplier in connected_suppliers:
        #         return True  # 找到终止条件，结束游戏
        return False

    # def agent_iter(self):
    #     while not all(self.terminations.values()) and not all(self.truncations.values()):
    #         yield self.agent_selection
    def agent_iter(self):
        yield self.agent_selection

    def next_agent(self):
        # 更新agent_selection
        agent_index = int(self.agent_selection.split('_')[-1])
        next_agent_index = (agent_index + 1) % self.num_agents
        self.agent_selection = f"agent_{next_agent_index}"
    def render(self, mode="human"):
        # 输出当前的节点供应商和奖励信息
        print(f"Current state: {self.nodes}")
        print(f"Current rewards: {self.rewards}")
        print(f"Current agent: {self.agent_selection}")

    def _all_nodes_assigned(self):
        """
        检查所有节点是否已经选择了供应商。
        """
        for node_data in self.nodes.values():
            if node_data['supplier'] == -1:
                return False
        return True

    def handle_trojan_nodes(self):
        """
        复制一份节点信息，遍历每个节点，将其当作木马源头，通过相同的供应商进行传播，
        直到传播无法继续为止。然后删除传播链上的所有节点，计算
        每个节点作为源头时的传播比例，并最终取平均。
        """
        total_nodes = len(self.nodes)
        spread_percentages = []

        for source_node_key in self.nodes.keys():
            # 创建节点的深度拷贝以避免影响原始数据
            nodes_copy = copy.deepcopy(self.nodes)
            infected_nodes = set()
            queue = []

            # 将源节点添加到感染集合和队列中
            infected_nodes.add(source_node_key)
            queue.append(source_node_key)

            while queue:
                current_node_key = queue.pop(0)
                current_node = nodes_copy[current_node_key]
                current_supplier = current_node['supplier']

                # 遍历当前节点的所有连接节点
                for neighbor_index in current_node['connections']:
                    neighbor_node_key = f"Node_{neighbor_index}"
                    neighbor_node = nodes_copy[neighbor_node_key]
                    neighbor_supplier = neighbor_node['supplier']

                    # 如果供应商相同且未被感染，则进行传播
                    if neighbor_supplier == current_supplier and neighbor_node_key not in infected_nodes:
                        infected_nodes.add(neighbor_node_key)
                        queue.append(neighbor_node_key)

            # 删除传播链上的所有节点
            for node_key in infected_nodes:
                del nodes_copy[node_key]

            # 计算传播比例
            spread_percentage = (len(infected_nodes) / total_nodes) * 100
            spread_percentages.append(spread_percentage)

        # 计算平均传播比例
        average_percentage = sum(spread_percentages) / len(spread_percentages)
        return average_percentage


