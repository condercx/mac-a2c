import argparse
import torch
import numpy as np
from train import train
from eval import eval
from network import MAC
import data_process
# 导入SimpleEnv类
from env import SimpleEnv

# 主程序入口
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="合作型游戏。")
    # 添加命令行参数，包括智能体数量、状态维度、动作维度、训练轮次等
    parser.add_argument("--num_agents", default=4, type=int)
    parser.add_argument("--num_nodes", default=49, type=int)
    parser.add_argument("--num_actions", default=49, type=int)
    parser.add_argument("--num_episode", default=10000, type=int)
    parser.add_argument("--lr_policy", default=0.001, type=float)
    parser.add_argument("--lr_value", default=0.001, type=float)
    parser.add_argument("--output_dir", default="output", type=str)
    # 添加训练和评估的布尔参数，无需指定值，仅表示开关
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use: 'cuda' or 'cpu'")  # **修改**
    # 解析命令行参数
    args = parser.parse_args()

    # 设置随机种子以确保实验可重复性
    torch.manual_seed(2)
    np.random.seed(2)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # **修改**
    print(f"Using device: {device}")  # **修改**

    # 初始化环境
    file_path = "network_topology.txt"  # 文件路径
    topology_matrix = data_process.read_network_topology(file_path)
    node_relationships = data_process.get_node_relationships(topology_matrix)
    env = SimpleEnv(num_agents=args.num_agents, node_data=node_relationships)
    # 初始化中心控制器
    central_controller = MAC(num_agents=args.num_agents, num_states=args.num_nodes, num_actions=args.num_actions)
    central_controller.to(device)
    # 根据命令行参数执行训练或评估
    # if args.do_train:
    #     train(args, env, central_controller)
    # if args.do_eval:
    #     eval(args)
    train(args, env, central_controller, topology_matrix, node_relationships, device)
    # eval(args)