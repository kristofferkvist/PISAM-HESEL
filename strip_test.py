import numpy as np

class Fun():
    def __init__(self):
        self.active = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0]).astype(np.bool_)
        self.max_ind = 6
        self.vacant_indices = np.array([]).astype(np.int32)
        self.x = np.array([2.1, 15, 15, 1.2, 5.1, 1.9, 15, 15, 15])

    def strip(self):
        mi_old = self.max_ind
        mask = ~self.active[0:mi_old]
        inactive = np.sum(mask)
        self.max_ind = self.max_ind - inactive
        mi_new = self.max_ind
        self.x[0:mi_new] = self.x[0:mi_old][self.active[0:mi_old]]
        self.x[mi_new:mi_old] = 10
        self.active[0:mi_new] = True
        self.active[mi_new:mi_old] = False
        self.vacant_indices = np.array([]).astype(np.int32)


fun = Fun()
fun.strip()
print(fun.active)
print(fun.x)
