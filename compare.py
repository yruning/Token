import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

"""
data_name = 'SVAMP'
model_name = 'phi-2'
data = np.load(data_name + '_data/'+data_name +'_'+model_name + '_ss.npy')
assert data.shape[0] == 2  # Two methods
a_scores = data[0]
b_scores = data[1]
t_stat, p_val = ttest_rel(b_scores, a_scores)
print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4g}")
plt.plot(data1, label='inter')
plt.plot(data2, label='key')
plt.xlabel('Data Point Index')
plt.ylabel('Performance')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
"""

data_name = 'HH'
types = ['toxicity','identity_attack','threat']
st = types[2]
#data1 = np.load(data_name + '_data/'+data_name +'_qwen-3b_ss.npy')
#data2 = np.load(data_name + '_data/'+data_name +'_qwen-3b-tag_ss.npy')
#print(np.mean(data1),np.mean(data2))

data1 = np.load(data_name + '_data/'+data_name +'_phi-2_'+st+'.npy')
data2 = np.load(data_name + '_data/'+data_name +'_phi-2-tag_'+st+'.npy')


print(np.mean(data1[:,0]),np.mean(data1[:,1]),np.mean(data2[:,0]),np.mean(data2[:,1]))
"""
plt.plot(data1[:,0], label='b_with')
plt.plot(data1[:,1], label='b_wo')
plt.plot(data2[:,0], label='dpo_w')
plt.plot(data2[:,1], label='dpo_wo')
plt.xlabel('Data Point Index')
plt.ylabel('Performance')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
"""