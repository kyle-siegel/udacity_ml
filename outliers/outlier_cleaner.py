#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    import math
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """


    ### your code goes here
    data = [(ages[i], net_worths[i], math.fabs(predictions[i]-net_worths[i])) for i in range(0, len(predictions))]
    sorted_data = sorted(data, key=lambda d: d[2])
    cleaned_data = sorted_data[0:81]

    return cleaned_data

