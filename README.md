## 环境配置
### **1. 环境要求**
- **Python** 3.8+
- 需要以下库：
  - `torch`
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `networkx`


### **2. 数据准备**
在运行代码之前，需要提供一个网络拓扑描述文件（如 `network_topology.txt`）。运行 `data_process.py` 的 `main` 函数生成。

---

## **代码结构**
以下是项目主要文件的功能说明：

- **`rollout.py`**：
  - 用于存储环境与智能体交互数据，并支持将数据转换为 PyTorch 张量。
  
- **`train.py`**：
  - 负责模型的训练逻辑，包括策略网络和价值网络的优化。
  - 自动保存模型和生成训练曲线（奖励和损失）。

- **`data_process.py`**：
  - 处理网络拓扑数据，包括清理、绘制和分析。
  - 提供 `read_network_topology` 和 `draw_network` 函数，直观展示节点的连接情况。

- **`env.py`**：
  - 定义环境类 `SimpleEnv`，模拟网络节点分配任务。
  - 支持多智能体交互和自定义奖励机制。

- **`network.py`**：
  - 定义策略网络（`PolicyNet`）和价值网络（`ValueNet`）。
  - 提供动作掩码生成逻辑，确保智能体只选择有效动作。

- **`csvtopdf.py`**：
  - 将训练过程中保存的 CSV 数据（如损失值和奖励）转换为 PDF 图表。

- **`a2c_main.py`**：
  - 项目主入口，通过命令行参数控制训练过程。

---

## **输出文件说明**
- **模型文件**：
  - 训练过程中会保存至 `output/model.pt`，包含所有策略网络的参数。
- **日志与曲线**：
  - 训练奖励和损失曲线将分别保存为 `Training Progress.pdf` 和 `Value Loss over Episodes.pdf`。
- **CSV 文件**：
  - 训练数据将记录在 `olg_lst.csv` 和 `value_loss_data.csv` 中。

---


## **使用示例**

运行以下命令训练：
```bash
python a2c_main.py
```
运行以下命令将csv结果转换为pdf格式的图：
```bash
python csvtopdf.py
```
---