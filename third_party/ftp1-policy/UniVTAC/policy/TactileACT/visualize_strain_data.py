import numpy as np
import matplotlib.pyplot as plt

strain_data_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_both_20/run_data/run_9/gelsight_strains.npy"



strain_data = np.load(strain_data_path)

print(strain_data.shape) # 156, 3
plt.figure()
plt.plot(strain_data[:,0], label="depth")
plt.plot(strain_data[:,1], label="x")
plt.plot(strain_data[:,2], label="y")
plt.legend()
plt.show()


