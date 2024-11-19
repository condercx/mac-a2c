import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取网络拓扑文件
def read_network_topology(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 去除每行的换行符，并将每行转为list[int]
        topology_matrix = [list(map(int, line.strip())) for line in lines]
    return np.array(topology_matrix)

# 获取每个节点的连接关系
def get_node_relationships(topology_matrix):
    num_nodes = topology_matrix.shape[0]
    node_relationships = {}

    for node in range(num_nodes):
        # 获取当前节点的供应商（可随机分配或根据需求修改逻辑）
        supplier = -1  # 初始供应商为-1
        # supplier = random.choice([-1, 0, 1])

        # 查找与当前节点相连的所有节点
        connected_nodes = [i for i in range(num_nodes) if topology_matrix[node][i] == 1]
        
        node_relationships[f"Node_{node}"] = {
            "supplier": supplier,
            "connections": connected_nodes
        }
    
    return node_relationships

def draw_network(topology_matrix, node_relationships):
    G = nx.Graph()
    
    # 添加节点及其属性（供应商）
    for node, info in node_relationships.items():
        G.add_node(node.replace("Node_", ""), supplier=info['supplier'])

    # 添加边（连接关系）
    num_nodes = topology_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if topology_matrix[i][j] == 1:
                G.add_edge(f"{i}", f"{j}")  # 修改这里

    # 获取节点的供应商属性
    suppliers = [info['supplier'] for info in node_relationships.values()]
    
    # 为每个供应商分配不同的颜色
    supplier_colors = {-1: 'gray', 0: 'green', 1: 'blue', 2: 'red', 3: 'orange', 4: 'purple', 5: 'brown'}
    node_colors = [supplier_colors[supplier] for supplier in suppliers]

    # 使用 grid_layout 实现矩阵排列
    grid_size = int(np.ceil(np.sqrt(num_nodes)))  # 确保矩阵能够容纳所有节点
    pos = {}
    for i in range(num_nodes):
        pos[f"{i}"] = (i % grid_size, grid_size - (i // grid_size))  # 修改这里

    # 绘制网络
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_color='white')

    # 添加图例
    legend_labels = {supplier: plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
                     for supplier, color in supplier_colors.items()}
    plt.legend(legend_labels.values(), legend_labels.keys(), title="Suppliers")
    
    # 显示图像
    plt.show()

def clean_adjacency_matrix(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 移除每行的空格，然后写入新的文件中
            cleaned_line = line.strip().replace(" ", "")
            outfile.write(cleaned_line + '\n')
# 主函数：读取文件并输出每个节点的关系和供应商
def main():
    clean_adjacency_matrix("240.txt", "network_topology.txt")
    file_path = "network_topology.txt"  # 文件路径
    topology_matrix = read_network_topology(file_path)
    node_relationships = get_node_relationships(topology_matrix)
    print(node_relationships)
    # 输出每个节点的供应商和连接关系
    # for node, info in node_relationships.items():
    #     print(f"{node}: Supplier = {info['supplier']}, Connections = {info['connections']}")
    # 绘制网络图
    draw_network(topology_matrix, node_relationships)

if __name__ == "__main__":
    main()
