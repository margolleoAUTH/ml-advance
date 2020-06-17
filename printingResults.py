import os
import graphviz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn import tree
from costcla.metrics import cost_loss, savings_score
from pdpbox import pdp


os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin/"


def print_multi_label_distribution(data_array):
    print("_______________________________")
    print("multi_label_distribution")
    distribution = []
    for i in range(0, len(data_array[0])):
        print(sum(data_array[:, i]))
        distribution.append(["Label" + str(i), sum(data_array[:, i])])
    df_stats = pd.DataFrame(distribution, columns=["labels", "number_of_instances"])
    df_stats.plot(x="labels", y="number_of_instances", kind="bar", legend=False, figsize=(8, 5))
    plt.title("Initial Data Distribution" + " Number of instances per label")
    plt.ylabel("# of instances", fontsize=8)
    plt.xlabel("labels", fontsize=8)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5)
    plt.savefig("multi_label_initial_data_distribution.pdf")
    print("_______________________________")


def print_multi_instance_distribution(data_array):
    print("_______________________________")
    print("multi_instance_distribution")
    df_stats = pd.DataFrame(data_array)
    df_label_stats = pd.DataFrame(data_array).groupby(df_stats.columns[1]).agg(["count"])
    values = df_label_stats.values
    indexes = df_label_stats.index.values
    label_distribution = "Class " + str(indexes[0]) + ": " + str(values[0][0]) + " - Class " + str(indexes[1]) + ": " + str(values[1][0])
    df_stats = pd.DataFrame(data_array).groupby(df_stats.columns[0]).agg(["count"])
    values = pd.DataFrame(df_stats.values)
    indexes = pd.DataFrame(df_stats.index.values)
    df_stats = indexes.join(values, how='left', lsuffix='_left', rsuffix='_right')
    df_stats.columns = ["bags", "number_of_instances"]
    df_stats.plot(x="bags", y="number_of_instances", kind="bar", legend=False, figsize=(8, 5))
    plt.title("Initial Data Distribution" + " Number of instances per bag" + "\n" + label_distribution)
    plt.ylabel("# of instances", fontsize=8)
    plt.xlabel("bags", fontsize=8)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5)
    plt.savefig("multi_instance_initial_data_distribution.pdf")
    print("_______________________________")


def print_multi_label_metrics(y_test, prediction, ranking=False):
    print("_______________________________")
    print("multi_label_metrics")
    decimals = 2
    print("hamming_loss %: ", round(metrics.hamming_loss(y_test, prediction) * 100, decimals))
    print("accuracy_score %: ", round((1 - metrics.hamming_loss(y_test, prediction)) * 100, decimals))
    if ranking:
        print("label_ranking_loss: ", round(metrics.label_ranking_loss(y_test, prediction), decimals))
        print("coverage_error: ", round(metrics.coverage_error(y_test, prediction), decimals))
        print("average_precision_score %: ", round(metrics.average_precision_score(y_test, prediction) * 100, decimals))
    print("subset accuracy_score %: ", round(metrics.accuracy_score(y_test, prediction) * 100, decimals))
    print("precision_score macro %: ", round(metrics.precision_score(y_test, prediction, average="macro") * 100, decimals))
    print("precision_score micro %: ", round(metrics.precision_score(y_test, prediction, average="micro") * 100, decimals))
    print("recall_score macro %: ", round(metrics.recall_score(y_test, prediction, average="macro") * 100, decimals))
    print("recall_score micro %: ", round(metrics.recall_score(y_test, prediction, average="micro") * 100, decimals))
    print("f1_score macro %: ", round(metrics.f1_score(y_test, prediction, average="macro") * 100, decimals))
    print("f1_score micro %: ", round(metrics.f1_score(y_test, prediction, average="micro") * 100, decimals))
    print("_______________________________")


def print_multi_instance_metrics(y_test, prediction, cost_matrix):
    print("_______________________________")
    print("multi_instance_metrics")
    decimals = 2
    print("accuracy_score %: ", round(metrics.accuracy_score(y_test, prediction) * 100, decimals))
    print("precision_score macro %: ", round(metrics.precision_score(y_test, prediction, average="macro") * 100, decimals))
    print("precision_score micro %: ", round(metrics.precision_score(y_test, prediction, average="micro") * 100, decimals))
    print("recall_score macro %: ", round(metrics.recall_score(y_test, prediction, average="macro") * 100, decimals))
    print("recall_score micro %: ", round(metrics.recall_score(y_test, prediction, average="micro") * 100, decimals))
    print("f1_score macro %: ", round(metrics.f1_score(y_test, prediction, average="macro") * 100, decimals))
    print("f1_score micro %: ", round(metrics.f1_score(y_test, prediction, average="micro") * 100, decimals))
    print("Cost Loss: " + str(round(cost_loss(y_test, np.array(prediction), cost_matrix), decimals)))
    print("Cost Savings Score: " + str(round(savings_score(y_test, np.array(prediction), cost_matrix), decimals)))
    print("_______________________________")


def plot_partial_dependence(title, classifier, coefficients, data, feature_names):
    print("_______________________________")
    print("plot_partial_dependence")
    feature_values_aggregation = []
    df = pd.DataFrame(coefficients)
    for column in df.columns:
        feature_values_aggregation.append(df[column].abs().sum())

    index = feature_values_aggregation.index(max(feature_values_aggregation))
    feature = feature_names[index]
    df = pd.DataFrame(data, columns=feature_names)
    pdp_goals = pdp.pdp_isolate(model=classifier, dataset=df, model_features=df.columns, feature=feature)
    pdp.pdp_plot(pdp_goals, feature)
    plt.title(title)
    plt.savefig(title + "_power_set_plot_partial_dependence.pdf")
    print("_______________________________")


def plot_trees_graph(title, model, tree_class_names, tree_feature_names, render):
    print("_______________________________")
    print("plot_trees_graph")
    dot_data = tree.export_graphviz(model,
                                    out_file=None,
                                    class_names=tree_class_names,
                                    feature_names=tree_feature_names,
                                    filled=True, rounded=True)

    dot_data = dot_data.replace("digraph Tree {", "digraph Tree {\nattr [label=\"" + title + "\"] ;", 1)
    graph = graphviz.Source(dot_data)
    graph.render(render)
    weights = model.feature_importances_
    model_weights = pd.DataFrame({"features": list(tree_feature_names), "weights": list(weights)})
    model_weights = model_weights[model_weights['weights'] > 0]
    model_weights = model_weights.sort_values(by="weights", ascending=False)
    plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor="w", edgecolor="k")
    sns.barplot(x="weights", y="features", data=model_weights)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.savefig(title + "_decision_tree_model_feature_importance.pdf")
    print("_______________________________")
