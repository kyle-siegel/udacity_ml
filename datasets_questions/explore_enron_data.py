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

enron_data = pickle.load(open("/Users/kylesiegel/Development/udacity/ud120-projects/final_project/final_project_dataset.pkl", "r"))
print len(enron_data.keys())
k = enron_data.keys()
print len(enron_data[k[0]])
print len([enron_data[i]["poi"] for i in enron_data.keys() if enron_data[i]["poi"] == 1])
print enron_data["PRENTICE JAMES"]['total_stock_value']
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']
ppl = ['LAY KENNETH L', 'SKILLING JEFFREY K', 'FASTOW ANDREW S']
print max([enron_data[i]['total_payments'] for i in ppl])
print len([enron_data[i]['salary'] for i in enron_data.keys() if enron_data[i]['salary'] != 'NaN'])
print len([enron_data[i]['email_address'] for i in enron_data.keys() if enron_data[i]['email_address'] != 'NaN'])
print len([enron_data[i]['total_payments'] for i in enron_data.keys() if enron_data[i]['total_payments'] == 'NaN'])

poi_list = [i for i in enron_data.keys() if enron_data[i]['poi'] == 1]
print 'Total POI:', len(poi_list)
print 'Total POI with NaN for Total Payments:', len([i for i in poi_list if enron_data[i]['total_payments'] == 'NaN'])