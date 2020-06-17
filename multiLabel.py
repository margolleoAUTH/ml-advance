import numpy as np
import sklearn.metrics as metrics
from statistics import mean
from printingResults import print_multi_label_metrics, plot_partial_dependence, plot_trees_graph
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.ensemble import RakelD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from bpmll.bpmll import bp_mll_loss
from sklearn.svm import SVC


def multi_label_class_expand_grouping(X, y):
    X_expanded = []
    y_expanded = []
    groups = []
    for i, labels in enumerate(y):
        if 1 in labels:
            for j, label in enumerate(labels):
                if label == 1:
                    X_expanded.append(X[i])
                    y_expanded.append(j)
                    groups.append(i)
        else:
            groups.append(i)
    return X_expanded, y_expanded, groups


def label_feature_expand(index, length):
    label_feature = np.zeros(length)
    label_feature[index] = 1
    label_feature = np.array2string(label_feature).replace(".", "").replace("[", "").replace("]", "").replace(" ", "")
    label_feature = float(label_feature)
    return label_feature


def multi_label_class_expand(X, y, scale_max, scale_min):
    class_distribution = [3.25, 5.25, 4.25, 1.25, 0.25, 2.25]
    X_expanded = []
    y_expanded = []
    for i, labels in enumerate(y):
        labels_length = len(labels)
        for j, label in enumerate(labels):
            temp = X[i].tolist()
            scale_value = class_distribution[j] / labels_length
            std = (scale_value - scale_min) / (scale_max - scale_min)
            scale_value = std * (1 - (-1)) + (-1)
            temp.append(scale_value)
            X_expanded.append(temp)
            if label == 1:
                y_expanded.append(1)
            else:
                y_expanded.append(-1)
    return X_expanded, y_expanded


def binary_relevance_simulation(x_train, x_test, y_train, y_test, parameters, cv):
    print("\n_______________________________")
    print("binary_relevance_simulation\n")
    classifier = GridSearchCV(BinaryRelevance(require_dense=[True, True]), parameters, cv=cv)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test).toarray()
    print(classifier.best_params_)
    print_multi_label_metrics(y_test, prediction)
    print("\n_______________________________\n")


def label_ranking_simulation(x_train, x_test, y_train, y_test, cv, calibration=0.5):
    print("\n_______________________________")
    print("label_ranking_simulation\n")

    x_train_expand, y_train_expand, groups = multi_label_class_expand_grouping(x_train, y_train)
    x_test_expand,  y_test_expand, groups = multi_label_class_expand_grouping(x_test, y_test)

    parametersLG = [
        {
            "estimator__C": [0.1, 1, 10],
            "estimator__class_weight": [None, "balanced"]
        }
    ]
    parametersSVC = [
        {
            "estimator__C": [0.1, 1, 10],
            "estimator__class_weight": [None, "balanced"]
        }
    ]

    classifier = GridSearchCV(OneVsOneClassifier(LogisticRegression(multi_class="auto", solver="liblinear", max_iter=5000)), parametersLG, cv=cv)
    classifier.fit(x_train_expand, y_train_expand)
    ovoSVC = GridSearchCV(OneVsOneClassifier(SVC(gamma="auto", kernel="linear", decision_function_shape="ovo", probability=True)), parametersSVC, cv=cv)
    ovoSVC.fit(x_train_expand, y_train_expand)
    if ovoSVC.best_score_ > classifier.best_score_:
        classifier = ovoSVC

    decision_function = classifier.decision_function(x_test_expand)
    x_test_expand = []
    distinct = []
    for i, row in enumerate(decision_function):
        if groups[i] not in distinct:
            distinct.append(groups[i])
            mean_value = mean(row) * calibration
            x_test_expand.append([])
            for cell in row:
                if cell > mean_value:
                    x_test_expand[len(distinct) - 1].append(1)
                else:
                    x_test_expand[len(distinct) - 1].append(0)
    print(classifier.best_params_)
    print_multi_label_metrics(y_test, x_test_expand)
    print("\n_______________________________\n")


def label_power_set_simulation(data, x_train, x_test, y_train, y_test, parameters, cv, feature_names):
    print("\n_______________________________")
    print("label_power_set_simulation\n")
    classifier = GridSearchCV(LabelPowerset(require_dense=[True, True]), parameters, cv=cv)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test).toarray()
    print(classifier.best_params_)
    print_multi_label_metrics(y_test, prediction)
    plot_partial_dependence("label_power_set", classifier.best_estimator_.classifier, classifier.best_estimator_.classifier.coef_, data.toarray(), feature_names)
    print("\n_______________________________\n")


def random_label_set_simulation(k, x_train, x_test, y_train, y_test, parameters, cv):
    print("\n_______________________________")
    print("random_label_set_simulation\n")
    classifier = GridSearchCV(RakelD(base_classifier_require_dense=[True, True], labelset_size=k), parameters, cv=cv)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test).toarray()
    print_multi_label_metrics(y_test, prediction)
    print("\n_______________________________\n")


def classifier_chain_simulation(x_train, x_test, y_train, y_test, parameters, cv):
    print("\n_______________________________")
    print("classifier_chain_simulation\n")
    classifier = GridSearchCV(ClassifierChain(require_dense=[True, True]), parameters, cv=cv)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test).toarray()
    print(classifier.best_params_)
    print_multi_label_metrics(y_test, prediction)
    print("\n_______________________________\n")


def stacked_binary_relevance_simulation(x_train, x_test, y_train, y_test, parameters, cv, feature_names, class_names):
    print("\n_______________________________")
    print("stacked_binary_relevance_simulation\n")
    classifier = GridSearchCV(BinaryRelevance(require_dense=[True, True]), parameters, cv=cv)
    classifier.fit(x_train, y_train)

    prediction = classifier.predict_proba(x_train)
    x_train_expanded = []
    for i, probability in enumerate(prediction):
        x_train_expanded.append(np.concatenate((x_train[i], np.asarray(probability.data[0], dtype=np.float32))))

    prediction = classifier.predict_proba(x_test)
    x_test_expanded = []
    for i, probability in enumerate(prediction):
        x_test_expanded.append(np.concatenate((x_test[i], np.asarray(probability.data[0], dtype=np.float32))))

    classifier.fit(x_train_expanded, y_train)
    prediction = classifier.predict(x_test_expanded).toarray()
    y_predicted = prediction
    print(classifier.best_params_)
    print_multi_label_metrics(y_test, prediction)
    print("\n_______________________________\n")
    new_x_train = x_train_expanded
    new_y_train = classifier.predict(x_train_expanded)
    for i in range(0, len(class_names)):
        feature_names.append("probability_" + str(i))
    parameters = {"criterion": ("gini", "entropy"), "splitter": ("best", "random"), "class_weight": (None, "balanced"), "max_depth": (1, 5)}
    classifier = GridSearchCV(DecisionTreeClassifier(), parameters, cv=cv)
    classifier.fit(new_x_train, new_y_train.toarray())
    prediction = classifier.predict(x_test_expanded)
    print(classifier.best_params_)
    print_multi_label_metrics(y_test, prediction)
    print("Fidelity %: ", round(metrics.accuracy_score(y_predicted, prediction) * 100, 2))
    folder_name = "stacked_binary_relevance_plot\\decision_tree_model"
    plot_trees_graph("stacked_binary_relevance", classifier.best_estimator_, class_names, feature_names, folder_name)
    print("\n_______________________________\n")


def ada_boost_mh_simulation(data, x_train, x_test, y_train, y_test, cv, feature_names, max, min):
    print("\n_______________________________")
    print("ada_boost_mh_simulation\n")
    x_train_expand, y_train_expand = multi_label_class_expand(x_train, y_train, max, min)
    x_test_expand, y_test_expand = multi_label_class_expand(x_test, y_test, max, min)
    feature_names.append("Label_Indicator")
    parameters = {"n_estimators": (50, 75, 100), "learning_rate": (1, 2)}
    classifier = GridSearchCV(AdaBoostClassifier(LogisticRegression(multi_class="auto", solver="liblinear", max_iter=5000)), parameters, cv=cv)
    classifier.fit(x_train_expand, y_train_expand)
    adaSVC = GridSearchCV(AdaBoostClassifier(SVC(gamma="auto", kernel="linear", decision_function_shape="ovo", probability=True)), parameters, cv=cv)
    adaSVC.fit(x_train_expand, y_train_expand)
    if adaSVC.best_score_ > classifier.best_score_:
        classifier = adaSVC
    prediction = classifier.predict(x_test_expand)
    print(classifier.best_params_)
    print_multi_label_metrics(y_test_expand, prediction, False)
    best_estimator = classifier.best_estimator_
    coefficients = []
    for clf, w in zip(best_estimator.estimators_, best_estimator.estimator_weights_):
        coefficients.append(clf.coef_ * w)
    coefficients = np.array(coefficients).mean(axis=0)
    data = data.toarray()
    data_placeholder = np.zeros((len(data), len(data[0]) + 1))
    data_placeholder[:, :-1] = data
    plot_partial_dependence("ada_boost_mh", best_estimator, coefficients, data_placeholder, feature_names)
    print("\n_______________________________\n")


def multi_label_c45_simulation(x_train, x_test, y_train, y_test, cv, feature_names, class_names):
    print("\n_______________________________")
    print("multi_label_c45_simulation\n")
    parameters = {"criterion": ("gini", "entropy"), "splitter": ("best", "random"), "class_weight": (None, "balanced"), "max_depth": (1, 15)}
    classifier = GridSearchCV(DecisionTreeClassifier(), parameters, cv=cv)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print(classifier.best_params_)
    print_multi_label_metrics(y_test, prediction)
    folder_name = "multi_label_c45_simulation_plot\\decision_tree_model"
    plot_trees_graph("multi_label_c45", classifier.best_estimator_, class_names, feature_names, folder_name)
    print("\n_______________________________\n")


def bp_mll_simulation(x_train, x_test, y_train, y_test):
    print("\n_______________________________")
    print("bp_mll_simulation\n")
    rounding = 8
    x_train = np.round(x_train, rounding).astype(np.float32)
    x_test = np.round(x_test, rounding).astype(np.float32)
    y_train = np.round(y_train, rounding).astype(np.float32)
    y_test = np.round(y_test, rounding).astype(np.float32)

    classifier = Sequential()
    classifier.add(Dense(128, input_dim=x_train.shape[1], activation="relu", kernel_initializer="glorot_uniform"))
    classifier.add(Dense(64, activation="relu", kernel_initializer="glorot_uniform"))
    classifier.add(Dense(y_train.shape[1], activation="sigmoid", kernel_initializer="glorot_uniform"))
    classifier.compile(loss=bp_mll_loss, optimizer="adagrad", metrics=[])

    classifier.fit(x_train, y_train, epochs=10000)
    prediction = classifier.predict(x_test)
    prediction = np.where(prediction > 0.5, 1, 0)
    print_multi_label_metrics(y_test, prediction)
    print("\n_______________________________\n")
