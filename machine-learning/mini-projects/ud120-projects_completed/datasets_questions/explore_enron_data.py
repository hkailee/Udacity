#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

# how many subjects (i.e. unique persons)
print len(enron_data.items())

# how many features
print len(enron_data['TAYLOR MITCHELL S'].keys()) 

# how many Persons of interest (POIs)
import pandas as pd
df = pd.DataFrame.from_dict(enron_data).T
print df['poi'].value_counts()

# James Prentice total stock value
print df.at['PRENTICE JAMES', 'total_payments']

# Total payment for 3 persons
print df.at['SKILLING JEFFREY K', 'total_payments']
print df.at['FASTOW ANDREW S', 'total_payments']
print df.at['LAY KENNETH L', 'total_payments']

# How many people have quantified salary
print len(df[df.salary != 'NaN'])
print len(df[df.email_address != 'NaN'])

# percentage of 'NaN' for total_payments 
import numpy as np
print 1.0 * len(df[df.total_payments == 'NaN']) / len(df)

print 1.0 * len(df[(df.total_payments == 'NaN') &
                   (df.poi == True)]) / len(df[df.poi == True])

print len(df[df.total_payments == 'NaN']) 
print df.columns