import numpy as np
from itertools import groupby
from operator import itemgetter
from multiInstanceUtils import calibration_simulation, \
    samples_weights_simulation, \
    class_weights_simulation, \
    sampling_simulation, \
    hausdorff_distance, \
    ckkn_sampling_simulation
from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def experiment_hanlder(x_training, x_testing, y_training, y_testing, feature_names=None, class_names=None):
    malignant_cost = 4
    benign_cost = 1
    fp = np.full((len(y_testing), 1), malignant_cost)
    fn = np.full((len(y_testing), 1), benign_cost)
    tp_tn = np.zeros((len(y_testing), 1))
    cost_matrix = np.hstack((fp, fn, tp_tn, tp_tn))

    sample_weight = np.zeros(len(y_training))
    sample_weight[np.where(np.array(y_training) == 2)] = malignant_cost
    sample_weight[np.where(np.array(y_training) == 1)] = benign_cost

    classifierLg = LogisticRegression(multi_class="auto", solver="liblinear", max_iter=5000)
    classifierRf = RandomForestClassifier(n_estimators=100)

    sampling_simulation(x_training, x_testing, y_training, y_testing, classifierLg, None, cost_matrix, malignant_cost,
                        benign_cost, feature_names, class_names, "LogisticRegression_")
    sampling_simulation(x_training, x_testing, y_training, y_testing, classifierRf, None, cost_matrix, malignant_cost,
                        benign_cost, feature_names, class_names, "RandomForestClassifier_")

    calibration_simulation(x_training, x_testing, y_training, y_testing, classifierLg, "sigmoid", cost_matrix)
    calibration_simulation(x_training, x_testing, y_training, y_testing, classifierLg, "isotonic", cost_matrix)
    samples_weights_simulation(x_training, x_testing, y_training, y_testing, classifierLg, sample_weight, cost_matrix)
    sampling_simulation(x_training, x_testing, y_training, y_testing, classifierLg, "undersampling", cost_matrix,
                        malignant_cost, benign_cost)
    sampling_simulation(x_training, x_testing, y_training, y_testing, classifierLg, "oversampling", cost_matrix,
                        malignant_cost, benign_cost)

    calibration_simulation(x_training, x_testing, y_training, y_testing, classifierRf, "sigmoid", cost_matrix)
    calibration_simulation(x_training, x_testing, y_training, y_testing, classifierRf, "isotonic", cost_matrix)
    samples_weights_simulation(x_training, x_testing, y_training, y_testing, classifierRf, sample_weight, cost_matrix)
    sampling_simulation(x_training, x_testing, y_training, y_testing, classifierRf, "undersampling", cost_matrix,
                        malignant_cost, benign_cost)
    sampling_simulation(x_training, x_testing, y_training, y_testing, classifierRf, "oversampling", cost_matrix,
                        malignant_cost, benign_cost)

    classifierLg = LogisticRegression(multi_class="auto", class_weight={1: malignant_cost, 2: benign_cost}, solver="liblinear", max_iter=5000)
    class_weights_simulation(x_training, x_testing, y_training, y_testing, classifierLg, cost_matrix)

    classifierRf = RandomForestClassifier(class_weight={1: malignant_cost, 2: benign_cost}, n_estimators=100)
    class_weights_simulation(x_training, x_testing, y_training, y_testing, classifierRf, cost_matrix)


def kmeans_simulation(data_X, data_mapping, n_clusters):
    print("\n_______________________________")
    print("kmeans_simulation\n")
    feature_names = []
    for i in range(0, n_clusters):
        feature_names.append("KAttr" + str(i))

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_X)
    group_column_index = len(data_mapping[0]) - 2
    bags_instances_grouped = groupby(data_mapping, key=itemgetter(group_column_index))
    data_X = []
    data_y = []

    for i in bags_instances_grouped:
        instances_group = list(i[1])
        features = np.zeros(n_clusters)
        for j in instances_group:
            instance = [j[:-2]]
            clusters = kmeans.predict(instance)
            for k in clusters:
                features[k] += 1
        data_y.append(instances_group[0][-1])
        data_X.append(features)

    x_training, x_testing, y_training, y_testing = train_test_split(data_X, data_y, test_size=0.25)
    minMaxScaler = MinMaxScaler(feature_range=(0, 1))
    x_training = minMaxScaler.fit_transform(x_training)
    x_testing = minMaxScaler.transform(x_testing)
    experiment_hanlder(x_training, x_testing, y_training, y_testing, feature_names, ["benign", "malignant"])
    print("\n_______________________________\n")


def kmedoids_simulation(data_X, data_mapping, n_clusters):
    print("\n_______________________________")
    print("kmedoids_simulation\n")
    kmedoids = KMedoids(n_clusters=n_clusters)
    kmedoids_labels = kmedoids.fit(data_X).labels_
    kmedoids_bags_mapping = np.zeros((len(data_X[:, 0]), len(data_X[0]) + 1))
    for i, row in enumerate(data_X):
        for j, cell in enumerate(row):
            kmedoids_bags_mapping[i][j] = cell
        kmedoids_bags_mapping[i][len(data_X[0])] = int(kmedoids_labels[i])

    group_column_index = len(data_mapping[0]) - 2
    bags_instances_grouped = groupby(data_mapping, key=itemgetter(group_column_index))
    group_column_index = len(data_X[0])
    kmedoids_bags_mapping = sorted(kmedoids_bags_mapping, key=itemgetter(group_column_index))
    data_X = []
    data_y = []
    for i in bags_instances_grouped:
        instances_group = list(i[1])
        label = instances_group[0][-1]
        instances_group = [item[:-2] for item in instances_group]
        features = []
        kmedoids_bags = groupby(kmedoids_bags_mapping, key=itemgetter(group_column_index))
        for j in kmedoids_bags:
            instances_cluster = list(j[1])
            instances_cluster = [item[:-1] for item in instances_cluster]
            feature = hausdorff_distance(instances_group, instances_cluster, decimals=2)
            features.append(feature)
        data_X.append(features)
        data_y.append(label)

    x_training, x_testing, y_training, y_testing = train_test_split(data_X, data_y, test_size=0.25)
    minMaxScaler = MinMaxScaler(feature_range=(0, 1))
    x_training = minMaxScaler.fit_transform(x_training)
    x_testing = minMaxScaler.transform(x_testing)
    experiment_hanlder(x_training, x_testing, y_training, y_testing)
    print("\n_______________________________\n")


def cknn_simulation(data_mapping, references, citers):
    print("\n_______________________________")
    print("cknn_simulation\n")
    group_column_index = len(data_mapping[0]) - 2
    bags_instances_grouped = groupby(data_mapping, key=itemgetter(group_column_index))
    group_list = []
    labels_length = int(data_mapping[len(data_mapping) - 1][group_column_index])
    labels = np.zeros(labels_length)
    for index, i in enumerate(bags_instances_grouped):
        instances_group = list(i[1])
        label = int(instances_group[0][-1])
        instances_group = [item[:-2] for item in instances_group]
        group_list.append(np.array(instances_group))
        labels[int(i[0] - 1)] = label

    x_training, x_testing, y_training, y_testing = train_test_split(group_list, labels, test_size=0.25)

    malignant_cost = 4
    benign_cost = 1
    fp = np.full((len(y_testing), 1), malignant_cost)
    fn = np.full((len(y_testing), 1), benign_cost)
    tp_tn = np.zeros((len(y_testing), 1))
    cost_matrix = np.hstack((fp, fn, tp_tn, tp_tn))

    ckkn_sampling_simulation(x_training, x_testing, y_training, y_testing, references, citers, "undersampling", cost_matrix,
                             malignant_cost, benign_cost)
    ckkn_sampling_simulation(x_training, x_testing, y_training, y_testing, references, citers, "oversampling", cost_matrix,
                             malignant_cost, benign_cost)
    ckkn_sampling_simulation(x_training, x_testing, y_training, y_testing, references, citers, None, cost_matrix, malignant_cost,
                             benign_cost)
    print("\n_______________________________\n")
