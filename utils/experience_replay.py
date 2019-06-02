import numpy as np

# replay class that stores n size tuple of experiences
class ExpReplay(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size       

    def __len__(self):
        return len(self._storage)

    def add(self, s, a, r, next_s, done):
        data = (s, a, r, next_s, done)
        
        self._storage.append(data)
        storage_size = len(self._storage)
        if (storage_size >= self._maxsize):
            self._storage = self._storage[storage_size-self._maxsize:]
        
    def sample(self, batch_size):
        idx = np.random.randint(0, len(self._storage), size=batch_size)
        batch = np.array(self._storage)[idx]
        states, actions, rewards, next_states, isdone = [], [], [], [], []
        
        for s, a, r, ns, done in batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            isdone.append(done)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(isdone)