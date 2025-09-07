import matplotlib.pyplot as plt

# Número de épocas
epochs = list(range(1, 48))

# Valores coletados do log (até epoch 47)
train_auc  = [0.9034,0.8691,0.8977,0.8971,0.9030,0.9814,0.9054,0.9222,0.9029,0.9290,
              0.9101,0.9173,0.9121,0.8620,0.9131,0.9922,0.9144,0.8688,0.9133,0.9027,
              0.9086,0.9222,0.9203,0.8587,0.9219,0.9648,0.9251,0.8786,0.9195,0.9434,
              0.9314,0.9258,0.9188,0.8773,0.9188,0.9671,0.9247,0.9424,0.9258,0.9714,
              0.9311,0.9740,0.9366,0.9648,0.9281,0.9180,0.9398]

val_auc    = [0.8467,0.8459,0.8411,0.8434,0.8531,0.8492,0.8868,0.8860,0.8861,0.8860,
              0.8905,0.8929,0.9063,0.9069,0.8672,0.8682,0.8754,0.8789,0.8860,0.8852,
              0.8506,0.8510,0.8597,0.8600,0.8514,0.8520,0.8668,0.8627,0.8826,0.8839,
              0.8560,0.8587,0.8658,0.8651,0.8869,0.8818,0.8627,0.8664,0.8640,0.8613,
              0.8710,0.8711,0.8528,0.8565,0.8736,0.8712,0.8671]

train_bin  = [0.2416,0.2776,0.2546,0.2795,0.2452,0.1395,0.2420,0.2228,0.2461,0.2124,
              0.2343,0.2287,0.2324,0.2764,0.2346,0.1466,0.2304,0.2603,0.2284,0.2456,
              0.2343,0.2236,0.2218,0.3058,0.2207,0.1596,0.2166,0.2776,0.2266,0.1720,
              0.2104,0.2410,0.2271,0.2536,0.2236,0.1848,0.2192,0.2021,0.2138,0.1485,
              0.2092,0.1397,0.2017,0.1724,0.2120,0.2387,0.1933]

val_bin    = [0.3862,0.3927,0.3796,0.3820,0.4047,0.4070,0.3797,0.3883,0.3048,0.3027,
              0.2966,0.2956,0.2972,0.2959,0.3641,0.3626,0.4079,0.3997,0.3352,0.3397,
              0.3848,0.3834,0.3802,0.3753,0.3752,0.3789,0.4261,0.4346,0.3945,0.4001,
              0.4688,0.4690,0.4072,0.4130,0.3405,0.3439,0.4491,0.4395,0.3693,0.3782,
              0.3641,0.3698,0.4067,0.3998,0.4006,0.4091,0.4235]

# Como 'loss' == 'binary_crossentropy', podemos usar os mesmos valores
train_loss = train_bin
val_loss   = val_bin

# ====== PLOTS ======
plt.figure(figsize=(14,4))

# AUC
plt.subplot(1,3,1)
plt.plot(epochs, train_auc, label="Train AUC")
plt.plot(epochs, val_auc, label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("AUC por época")
plt.legend()

# Binary Crossentropy
plt.subplot(1,3,2)
plt.plot(epochs, train_bin, label="Train BinaryCross")
plt.plot(epochs, val_bin, label="Val BinaryCross")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy")
plt.title("Binary Crossentropy por época")
plt.legend()

# Loss
plt.subplot(1,3,3)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss por época")
plt.legend()

plt.tight_layout()
plt.show()
