def decay(frame, min_decay, no_decay_threshold):
    """Linear decay from 1 (frame 0) to min_decay (frame no_decay_threshold),
       min_decay thereafter"""
    return min_decay if (frame > no_decay_threshold)\
                     else (min_decay-1)*frame/no_decay_threshold +1

import numpy as np
print([decay(np.array(range(10)[i]), 0.1, 8) for i in range(10)])
