import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('value_loss_data.csv')
value_loss_lst = df['Value Loss'].tolist()
x_axis = np.arange(len(value_loss_lst))
plt.plot(x_axis, value_loss_lst)
plt.xlabel("Episode", fontsize=22)
plt.ylabel("Value Loss", fontsize=22)
plt.tick_params(axis='both', labelsize=18)
# plt.xticks(np.linspace(0, 6000, 7))
plt.axis([0, 6000, 0, 600])
plt.savefig("Value Loss over Episodes.pdf", bbox_inches="tight")
plt.close()

df = pd.read_csv('olg_lst.csv')
olg_lst = df['olg_lst'].tolist()
x_axis = np.arange(len(olg_lst))
plt.plot(x_axis, olg_lst)
plt.xlabel("Episode", fontsize=22)
plt.ylabel("OL(G) (%)", fontsize=22)
plt.tick_params(axis='both', labelsize=18)
# plt.xticks(np.linspace(0, 6000, 7))
plt.axis([0, 6000, 74, 92])
plt.savefig("Training Progress.pdf", bbox_inches="tight")
plt.close()