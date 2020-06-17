import numpy as np
import datetime
from printingResults import print_multi_label_distribution, print_multi_instance_distribution
from multiInstance import kmeans_simulation, kmedoids_simulation, cknn_simulation
from multiLabel import binary_relevance_simulation, \
    label_ranking_simulation, \
    label_power_set_simulation, \
    random_label_set_simulation, \
    classifier_chain_simulation, \
    stacked_binary_relevance_simulation, \
    ada_boost_mh_simulation, \
    multi_label_c45_simulation, \
    bp_mll_simulation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from skmultilearn.dataset import load_from_arff
from scipy.io import loadmat
from scipy.io import arff


# ===================================================================================================
# IMPORTANT NOTE!!!
# Unfortunately, our IDE had already installed some libraries before we started the implementation.
# This was a trap for us and as a result we did not maintain the requirement.txt file.
# So sorry! requirements.txt is DEPRECATED do not run pip install using this configuration file
# ===================================================================================================

# Extra Comments
#
# git config --replace-all user.name "margolleoAUTH"
# git config --replace-all user.email "margolleo@csd.auth.com"
# git commit --author="margolleoAUTH <margolleo@csd.auth.com>"


def multi_label():
    path_to_arff_file = "Scene.arff"
    label_count = 6
    label_location = "end"
    arff_file_is_sparse = False
    data_X, data_y = load_from_arff(
        path_to_arff_file,
        label_count=label_count,
        label_location=label_location,
        load_sparse=arff_file_is_sparse
    )
    data = arff.loadarff(path_to_arff_file)
    attributes = list(data[0][0].dtype.names)[0:294]
    labels = list(data[0][0].dtype.names)[294:300]
    print_multi_label_distribution(data_y.toarray())
    x_training, x_testing, y_training, y_testing = train_test_split(data_X, data_y, test_size=0.25)

    x_training = x_training.toarray()
    x_testing = x_testing.toarray()
    y_training = y_training.toarray()
    y_testing = y_testing.toarray()

    minMaxScaler = MinMaxScaler(feature_range=(-1, 1))
    x_training = minMaxScaler.fit_transform(x_training)
    x_testing = minMaxScaler.transform(x_testing)

    folds = 5
    k_labels = 7
    grid_search_parameters = [
        {
            "classifier": [LogisticRegression(multi_class="auto", solver="liblinear", max_iter=5000)],
            "classifier__C": [0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        },
        {
            "classifier": [SVC(gamma="auto", kernel="linear", decision_function_shape="ovo", probability=True)],
            "classifier__C": [0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    ]
    grid_search_parameters_random_label_set = [
        {
            "base_classifier": [LogisticRegression(multi_class="auto", solver="liblinear", max_iter=5000)],
            "base_classifier__C": [0.1, 1, 10],
            "base_classifier__class_weight": [None, "balanced"]
        },
        {
            "base_classifier": [SVC(gamma="auto", kernel="linear", decision_function_shape="ovo", probability=True)],
            "base_classifier__C": [0.1, 1, 10],
            "base_classifier__class_weight": [None, "balanced"]
        }
    ]
    bp_mll_simulation(x_training, x_testing, y_training, y_testing)
    multi_label_c45_simulation(x_training, x_testing, y_training, y_testing, folds, attributes, labels)
    label_ranking_simulation(x_training, x_testing, y_training, y_testing, folds, True)
    label_power_set_simulation(data_X, x_training, x_testing, y_training, y_testing, grid_search_parameters, folds, attributes)
    binary_relevance_simulation(x_training, x_testing, y_training, y_testing, grid_search_parameters, folds)
    random_label_set_simulation(k_labels, x_training, x_testing, y_training, y_testing, grid_search_parameters_random_label_set, folds)
    classifier_chain_simulation(x_training, x_testing, y_training, y_testing, grid_search_parameters, folds)
    stacked_binary_relevance_simulation(x_training, x_testing, y_training, y_testing, grid_search_parameters, folds, attributes, labels)
    ada_boost_mh_simulation(data_X, x_training, x_testing, y_training, y_testing, folds, attributes, max(minMaxScaler.data_range_), min(minMaxScaler.data_range_))

# ==================================================================================
# ==================================================================================
# ==================================================================================


def multi_instance():
    X = loadmat("ucsb_breast.mat")
    data = X["x"][0][0]["data"]
    nlab = X["x"][0][0]["nlab"]
    ident = X["x"][0][0]["ident"][0][0][1]
    data_X = np.zeros((len(data[:, 0]), len(data[0])))
    data_y = np.zeros(len(nlab[:, 0]))
    data_mapping = np.zeros((len(data[:, 0]), len(data[0]) + 2))
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            data_X[i][j] = cell
            data_mapping[i][j] = cell
        data_y[i] = ident[i][0]
        data_mapping[i][len(data_mapping[0]) - 2] = ident[i][0]
        data_mapping[i][len(data_mapping[0]) - 1] = nlab[i][0]
    start = len(data_mapping[0])-2
    end = len(data_mapping[0])
    distribution = data_mapping[:, start:end]
    print_multi_instance_distribution(distribution)
    kmeans_simulation(data_X, data_mapping, 3)
    cknn_simulation(data_mapping, 12, 14)
    kmedoids_simulation(data_X, data_mapping, 3)


if __name__ == "__main__":

    try:
        print("_______________________________")
        print(datetime.datetime.now().strftime("%D %H:%M:%S"))
        multi_label()
        multi_instance()
        print("_______________________________")
        print(datetime.datetime.now().strftime("%D %H:%M:%S"))

    except Exception as error:
        print("===================================================================================================")
        print("Error on_main: %s" % str(error))
        print("===================================================================================================")
