#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy as np
    import pandas as pd

    errors = np.abs(net_worths-predictions)

    top10p = np.percentile(errors, 90, axis=0)

    # create boolean value for each item
    bool_top10p_errors = errors < top10p

    # select data with True Boolean
    cleaned_data = np.stack((ages[bool_top10p_errors], 
                              net_worths[bool_top10p_errors], 
                              errors[bool_top10p_errors]), axis=1)

    return cleaned_data

