#!/usr/bin/python


def main():
    import sys
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    sys.path.append("/Users/kylesiegel/Development/udacity/ud120-projects/tools")

    from feature_format import featureFormat, targetFeatureSplit
    from tester import dump_classifier_and_data

    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi', 'fin_pca', 'from_ratio', 'to_ratio']  # You will need to use more features

    ### Load the dictionary containing the dataset
    with open("/Users/kylesiegel/Development/udacity/ud120-projects/final_project/final_project_dataset.pkl", "r") \
            as data_file:
        data_dict = pickle.load(data_file)

    ### Task 2: Remove outliers
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    del data_dict['TOTAL']

    data_dict = impute(data_dict,
                       ['bonus', 'total_payments', 'exercised_stock_options','from_poi_to_this_person', 'to_messages', 'from_this_person_to_poi', 'from_messages'])

    pca_keys = ['total_payments', 'exercised_stock_options', 'bonus']

    data_for_pca = []
    for key in pca_keys:
        data_for_pca.append([data_dict[name][key] for name in data_dict.keys()])

    print data_for_pca

    from sklearn.decomposition import PCA
    # data_for_pca = np.array(zip(x, y, z))
    pca = PCA(n_components=1)
    transformed_data = pca.fit_transform(np.array(zip(*data_for_pca)))

    k = data_dict.keys()
    for i in range(0, len(k)):
        data_dict[k[i]]['fin_pca'] = transformed_data[i]

    for i in range(0, len(k)):
        data_dict[k[i]]['from_ratio'] = float(data_dict[k[i]]['from_poi_to_this_person']) / \
                                        float(data_dict[k[i]]['to_messages'])
        data_dict[k[i]]['to_ratio'] = float(data_dict[k[i]]['from_this_person_to_poi']) / \
                                      float(data_dict[k[i]]['from_messages'])

    my_dataset = data_dict
    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys=True)



    labels, features = targetFeatureSplit(data)

    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    # Provided to give you a starting point. Try a variety of classifiers.
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    # from sklearn.ensemble import AdaBoostClassifier
    # clf = AdaBoostClassifier()



    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    # Example starting point. Try investigating other evaluation techniques!
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, features_list)
    return data_dict

def outlierCleaner(predictions, x, y):
    import math
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (x, y).
    """
    data = [(x[i], y[i], math.fabs(predictions[i]-y[i])) for i in range(0, len(predictions))]
    sorted_data = sorted(data, key=lambda d: d[2])
    cleaned_data = sorted_data[0:int(math.floor(len(x)*.9))]
    cleaned_data = [i[0:2] for i in cleaned_data]

    return cleaned_data

def impute(data_dict, keys):
    from sklearn.preprocessing import Imputer
    for key in keys:
        x = [data_dict[k][key] for k in data_dict.keys()]
        imp = Imputer(missing_values='NaN', strategy="mean", axis=1)
        imp.fit(x)
        x = imp.transform(x)[0]
        names = data_dict.keys()
        for j in range(0, len(data_dict.keys())):
            data_dict[names[j]][key] = x[j]

    return data_dict


if __name__ == "__main__":
    main()