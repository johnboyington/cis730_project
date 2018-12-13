import numpy as np


class Data_Container(object):
    """A container for data used to train our ANN."""
    def __init__(self, savename='data.npy', method='default'):
        self.data = []
        self.savename = savename
        self.method = method

    def log_step(self, timestep, action):
        """This method takes in a pysc2 timestep and an action index
        and stores info from it."""

        # the following is a default method for data storage
        if self.method == 'default':
            row = []
            fs = timestep.observation.feature_screen

            for screen in [fs.unit_hit_points, fs.unit_type]:
                row += list(screen.flatten())

            row.append(action)
            self.data.append(np.array(row))
            self.save_data()
            return
        
        elif self.method == 'compress':
            row = []
            fs = timestep.observation.feature_screen

            for map_type, screen in [('hp', fs.unit_hit_points), ('pid' ,fs.unit_type)]:
                screen = np.array(screen)
                # represent data as 31x31
                sizex = 31
                sizey = 31
                px = int(screen.shape[0] / sizex)
                py = int(screen.shape[1] / sizey)
                
                img = np.empty((sizex, sizey))
                
                # loop over data
                for i in range(sizex):
                    for j in range(sizey):
                        x = i*px
                        y = j*py
                        if map_type == 'hp':
                            img[i, j] = np.average(screen[x:x+px,y:y+py])
                        elif map_type == 'pid':
                            val = np.max(screen[x:x+px,y:y+py])
                            if val == 110:
                                val = 1
                            else:
                                val = 0
                            img[i, j] = val
                row += list(img.flatten())

            row.append(action)
            self.data.append(np.array(row))
            return

    def save_data(self):
        """A utility to store the data in a human readable numpy array."""
        try:
            data = np.load('data.npy')
            savedata = np.array(self.data)
            savedata = np.concatenate([data, savedata])
        except:
            savedata = np.array(self.data)
        np.save(self.savename, savedata)
        return
