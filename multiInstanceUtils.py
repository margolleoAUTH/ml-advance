import numpy as np
from ccknn import CKNN
from printingResults import print_multi_instance_metrics, plot_trees_graph
from scipy.spatial import distance
from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from costcla.models import BayesMinimumRiskClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics


def euclidean_distance(point_a, point_b):
    return distance.euclidean(point_a, point_b)


def h(set_a, set_b):
    return max((min(list(euclidean_distance(a, b) for b in set_b)) for a in set_a))


def hausdorff_distance(set_x, set_y, decimals=2):
    return round(max(h(set_x, set_y), h(set_y, set_x)), decimals)


def calibration_simulation(x_training, x_testing, y_training, y_testing, classifier, method, cost_matrix):
    print("\ncalibration_simulation\n")
    calibratedClassifier = CalibratedClassifierCV(classifier, method=method, cv=3)
    model = calibratedClassifier.fit(x_training, y_training)
    prob_testing = model.predict_proba(x_testing)
    bayesMinimumRiskClassifier = BayesMinimumRiskClassifier(calibration=False)
    prediction = bayesMinimumRiskClassifier.predict(prob_testing, cost_matrix)
    y_testing = (np.array(y_testing) != 1).astype(np.float)
    print_multi_instance_metrics(y_testing, prediction, cost_matrix)


def samples_weights_simulation(x_training, x_testing, y_training, y_testing, classifier, sample_weight, cost_matrix):
    print("\nsamples_weights_simulation\n")
    classifier.fit(x_training, y_training, sample_weight)
    prediction = classifier.predict(x_testing)
    print_multi_instance_metrics(y_testing, prediction, cost_matrix)


def class_weights_simulation(x_training, x_testing, y_training, y_testing, classifier, cost_matrix):
    print("\nclass_weights_simulation\n")
    classifier.fit(x_training, y_training)
    prediction = classifier.predict(x_testing)
    print_multi_instance_metrics(y_testing, prediction, cost_matrix)


def sampling_simulation(x_training, x_testing, y_training, y_testing, classifier, sample_type, cost_matrix, malignant_cost, benign_cost, feature_names=None, class_names=None, model_interpret=None):
    if sample_type is "undersampling":
        print("\nsampling_simulation__undersampling\n")
        malignant_count = len(np.where(np.array(y_training) == 2)[0])
        benign_count = int(round(malignant_count / malignant_cost, 0))
        sampler = RandomUnderSampler(sampling_strategy={1: benign_count, 2: malignant_count})
        x_training, y_training = sampler.fit_sample(x_training, y_training)
    elif sample_type is "oversampling":
        print("\nsampling_simulation__oversampling\n")
        benign_count = len(np.where(np.array(y_training) == 1)[0])
        malignant_count = int(round(benign_count * benign_cost, 0))
        sampler = RandomOverSampler(sampling_strategy={1: benign_count, 2: malignant_count})
        x_training, y_training = sampler.fit_sample(x_training, y_training)
    else:
        print("\nsampling_simulation__combination\n")
        factor = 3
        malignant_count = len(np.where(np.array(y_training) == 2)[0])
        benign_count = len(np.where(np.array(y_training) == 1)[0])
        benign_count = int(round(benign_count / factor, 0))
        sampler = RandomUnderSampler(sampling_strategy={1: benign_count, 2: malignant_count})
        x_resample, y_resample = sampler.fit_sample(x_training, y_training)

        malignant_count = int(round(benign_count * malignant_cost, 0))
        sampler = RandomOverSampler(sampling_strategy={1: benign_count, 2: malignant_count})
        x_training, y_training = sampler.fit_sample(x_resample, y_resample)

    classifier.fit(x_training, y_training)
    prediction = classifier.predict(x_testing)
    print_multi_instance_metrics(y_testing, prediction, cost_matrix)
    print("\n_______________________________\n")
    if sample_type is not "undersampling" and sample_type is not "oversampling" and feature_names is not None and class_names is not None and model_interpret is not None:
        y_predicted = prediction
        new_x_train = x_training
        new_y_train = classifier.predict(x_training)
        parameters = {"criterion": ("gini", "entropy"), "splitter": ("best", "random"), "class_weight": (None, "balanced"), "max_depth": (1, 5)}
        classifier = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5)
        classifier.fit(new_x_train, new_y_train)
        prediction = classifier.predict(x_testing)
        print(classifier.best_params_)
        print_multi_instance_metrics(y_testing, prediction, cost_matrix)
        print("Fidelity %: ", round(metrics.accuracy_score(y_predicted, prediction) * 100, 2))
        folder_name = model_interpret + "sampling_combination_plot\\decision_tree_model"
        plot_trees_graph(model_interpret + "sampling_simulation__combination", classifier.best_estimator_, class_names, feature_names, folder_name)


def ckkn_sampling_simulation(x_training, x_testing, y_training, y_testing, references, citers, sample_type, cost_matrix, malignant_cost, benign_cost):
    if sample_type is "undersampling":
        print("\nsampling_simulation__undersampling\n")
        malignant_count = len(np.where(np.array(y_training) == 2)[0])
        benign_count = int(round(malignant_count / malignant_cost, 0))
        sampler = RandomUnderSampler(sampling_strategy={1: benign_count, 2: malignant_count})
        x_training = np.asarray(x_training).reshape(-1, 1)
        x_training, y_training = sampler.fit_sample(x_training, y_training)
        x_training = [np.asarray(item[0]) for item in x_training]
    elif sample_type is "oversampling":
        print("\nsampling_simulation__oversampling\n")
        benign_count = len(np.where(np.array(y_training) == 1)[0])
        malignant_count = int(round(benign_count * benign_cost, 0))
        sampler = RandomOverSampler(sampling_strategy={1: benign_count, 2: malignant_count})
        x_training = np.asarray(x_training).reshape(-1, 1)
        x_training, y_training = sampler.fit_sample(x_training, y_training)
        x_training = [np.asarray(item[0]) for item in x_training]
    else:
        print("\nsampling_simulation__combination\n")
        factor = 3
        malignant_count = len(np.where(np.array(y_training) == 2)[0])
        benign_count = len(np.where(np.array(y_training) == 1)[0])
        benign_count = int(round(benign_count / factor, 0))
        sampler = RandomUnderSampler(sampling_strategy={1: benign_count, 2: malignant_count})
        x_training = np.asarray(x_training).reshape(-1, 1)
        x_resample, y_resample = sampler.fit_sample(x_training, y_training)
        x_training = [np.asarray(item[0]) for item in x_training]

        malignant_count = int(round(benign_count * malignant_cost, 0))
        sampler = RandomOverSampler(sampling_strategy={1: benign_count, 2: malignant_count})
        np.asarray(x_training).reshape(-1, 1)
        x_training, y_training = sampler.fit_sample(x_resample, y_resample)
        x_training = [np.asarray(item[0]) for item in x_training]

    classifier = CKNN()
    classifier.fit(x_training, y_training, references=references, citers=citers)
    prediction = classifier.predict(x_testing)
    print_multi_instance_metrics(y_testing, prediction, cost_matrix)
