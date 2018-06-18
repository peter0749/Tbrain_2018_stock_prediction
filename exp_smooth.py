import numpy as np
def exp_smooth(xs, alpha=0.7, dtype=np.float32):
    s = [xs[0]]
    for x in xs[1:]:
        s.append(alpha*x + (1.-alpha)*s[-1])
    return np.asarray(s, dtype=dtype)
