# (0, 1) to (-1, 1) scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in x and returns that x, scaled
       with a feature_range of values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max - min) + min

    return x

