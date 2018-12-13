import numpy as np
import matplotlib.pyplot as plt


def take_snapshot(obs):
    """A utility that grabs and stores an observation."""
    fs = obs.observation.feature_screen
    pid = fs.player_id
    np.save('snapshot.npy', pid)
    return

def inspect_observation(filename):
    """A utility to understand an observation."""
    hp = np.load(filename)
    print(hp[50][10:20])
    plt.figure(0)
    plt.imshow(hp)
    
    # represent data as 31x31
    sizex = 31
    sizey = 31
    px = int(hp.shape[0] / sizex)
    py = int(hp.shape[1] / sizey)
    
    img = np.empty((sizex, sizey))
    
    # loop over data
    for i in range(sizex):
        for j in range(sizey):
            x = i*px
            y = j*py
            val = np.max(hp[x:x+px,y:y+py])
            img[i, j] = val
    
    plt.figure(1)
    plt.imshow(img)
    plt.colorbar()
    print(set(img.flatten()))
    


if __name__ == '__main__':
    inspect_observation('snapshot.npy')