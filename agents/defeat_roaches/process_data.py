import numpy as np
import matplotlib.pyplot as plt
from transform_action import transform_action

data = np.load('data.npy')
print('length', len(data))
maps = data[:, :-1]
labels = data[:, -1]

d = {}
for da in data[0][961:-1]:
    if da:
        if da in d:
            d[da] += 1
        else:
            d[da] = 1


frame = 1500

hp = maps[frame][:961].reshape(31, 31)
pid = maps[frame][961:].reshape(31, 31)

plt.imshow(pid)
x, y = transform_action(int(labels[frame]))
print(x/4, y/4)