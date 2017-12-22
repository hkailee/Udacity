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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
exercised_stock_options = []
salary = []
for key1, val1 in enron_data.items():
    if not key1 == "TOTAL":
        for key2, val2 in val1.items():
            if key2 == "exercised_stock_options":
                if val2 != "NaN":
                    exercised_stock_options.append(val2)
            elif key2 == "salary":
                if not val2 == "NaN":
                    salary.append(val2)
                    
print "maximum value of exercised_stock_options:", max(exercised_stock_options)
print "minimum value of exercised_stock_options:", min(exercised_stock_options)                    
print "maximum value of salary:", max(salary)
print "minimum value of salary:", min(salary)
