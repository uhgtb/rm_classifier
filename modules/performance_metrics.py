import numpy as np
import pandas as pd
import math
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import cluster_performance_metrics as cpm


def confusion_matrix(true_val, pred_val, colnames=["predicted"], rownames=["true"], categories=None):
    """
    Create a confusion matrix from true and predicted values.
    If categories is None, it will be inferred from the unique values in true_val and pred_val.
    The confusion matrix will be a pandas DataFrame with the specified row and column names.
    Args:
        true_val (array-like): True labels.
        pred_val (array-like): Predicted labels.
        colnames (list): Column names for the confusion matrix.
        rownames (list): Row names for the confusion matrix.
        categories (list): List of categories to include in the confusion matrix.
    Returns:
        pd.DataFrame: Confusion matrix.
    """
    true_val = np.array(true_val).astype(str)
    pred_val = np.array(pred_val).astype(str)
    if categories is None:
        categories = np.unique(np.concatenate((true_val, pred_val)))
    true_val = pd.Categorical(true_val, categories=categories) # to make the matrix quadratic
    pred_val = pd.Categorical(pred_val, categories=categories) # to make the matrix quadratic
    return pd.crosstab(true_val, pred_val, colnames=colnames, rownames=rownames, dropna=False)

def confusion_matrix_from_dominant_cluster_label(true_val, cluster_labels, colnames=["predicted"], rownames=["true"], categories=None):
    """
    Create a confusion matrix from true and predicted values.
    The predicted values are inferred from the dominant cluster label for each true label.
    It can be especially used, if true labels are malfunction types and predicted values are cluster labels and each identified cluster is assigned to the malfunction type, which is most frequent in this cluster.
    Args:
        true_val (array-like): True labels.
        cluster_labels (array-like): Cluster labels.
        colnames (list): Column names for the confusion matrix.
        rownames (list): Row names for the confusion matrix.
        categories (list): List of categories to include in the confusion matrix.
    Returns:
        pd.DataFrame: Confusion matrix.
    """
    contingency_table = pd.crosstab(true_val, cluster_labels, colnames=colnames, rownames=rownames)

    main_true = contingency_table.idxmax(axis=0)
    pred_val, true_val=[],[]
    for i in contingency_table.index:
        for j in contingency_table.keys():
            val = contingency_table.loc[(i,j)]
            pred_val.extend(np.repeat(main_true[j], val))
            true_val.extend(np.repeat(i, val))
    return contingency_table, confusion_matrix(true_val, pred_val, colnames=colnames, rownames=rownames, categories=categories)

def prediction_performance(confusion_matrix, contingency_table=None, times = None, noise_label=-1, normal_label="normal", verbose=False):
    """
    Calculate main performance metrics from a confusion matrix and optionally a contingency_table.
    The confusion matrix should have true labels as rows and predicted labels as columns.
    If contingency_table is provided, it should be a confusion matrix with true labels as rows and cluster labels as columns.
    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix with true labels as rows and predicted labels as columns.
        contingency_table (pd.DataFrame): Contingency table with true labels as rows and cluster labels as columns.
        times (array-like): CPU and wall time for the prediction, if it shall be included in the performance summary.
        noise_label (str or int): Label used for noise points.
        normal_label (str or int): Label used for normal points.
        verbose (bool): If True, print warnings for missing labels.
    Returns:
        performance_params_summary (dict): Summary of performance metrics.
        detailed_performance (pd.DataFrame): Detailed performance metrics for each true label.
    """
    labels = {"normal": normal_label, "noise": noise_label}
    if str(noise_label)==str(normal_label):
        raise ValueError("Noise label and normal label cannot be the same")
    for label in ["noise", "normal"]:
        if label not in list(confusion_matrix.index) and label not in list(confusion_matrix.columns):
            if verbose:
                print(f"Warning: {label} label not found in confusion matrix, trying to replace it with {labels[label]}")
            confusion_matrix.index = confusion_matrix.index.astype(str).str.replace(str(labels[label]), label) # replace cluster labels with noise or normal
            confusion_matrix.columns = confusion_matrix.columns.astype(str).str.replace(str(labels[label]), label)
        elif label != labels[label]:
            if verbose:
                print(f"Warning: {label} label found in confusion matrix, but it is not the same as {labels[label]}, keeping label {label}")
        if label not in list(confusion_matrix.index):
            if verbose:
                print(f"Warning: {label} label not found in confusion matrix, adding it with 0 counts")
            confusion_matrix.loc[label, :] = 0
        if label not in list(confusion_matrix.columns):
            if verbose:
                print(f"Warning: {label} label not found in confusion matrix, adding it with 0 counts")
            confusion_matrix.loc[:, label] = 0
        if not isinstance(contingency_table, type(None)):
            if label not in list(contingency_table.index) and label not in list(contingency_table.columns):
                if verbose:
                    print(f"Warning: {label} label not found in contingency_table, trying to replace it with {labels[label]}")
                contingency_table.index = contingency_table.index.astype(str).str.replace(str(labels[label]), label) # replace cluster labels with noise or normal
                contingency_table.columns = contingency_table.columns.astype(str).str.replace(str(labels[label]), label)
            elif label != labels[label]:
                if verbose:
                    print(f"Warning: {label} label found in contingency_table, but it is not the same as {labels[label]}, keeping label {label}")
            if label not in list(contingency_table.index):
                if verbose:
                    print(f"Warning: {label} label not found in contingency_table, adding it with 0 counts")
                contingency_table.loc[label, :] = 0
            if label not in list(contingency_table.columns):
                if verbose:
                    print(f"Warning: {label} label not found in contingency_table, adding it with 0 counts")
                contingency_table.loc[:, label] = 0
                
    confusion_matrix = confusion_matrix.astype(int)
   
    
    performance_params=pd.DataFrame(index=confusion_matrix.index)
    performance_params["true_malfunction_pred_noise"] = confusion_matrix.loc[:,"noise"]/(confusion_matrix.sum(axis=1)+1e-12) # fraction of the true malfunction that are predicted as noise
    performance_params["true_noise_pred_malfunction"] = np.array(0)
    performance_params["n_true"] = confusion_matrix.sum(axis=1)
    performance_params["n_pred"] = confusion_matrix.sum(axis=0)
    def diag(cm):
        return np.array([cm[key][key] for key in cm.index])
    performance_params["efficiency"] = diag(confusion_matrix)/(confusion_matrix.sum(axis=1)+1e-12)
    performance_params["purity"] = diag(confusion_matrix)/(confusion_matrix.sum(axis=0)+1e-12)
    performance_params["fake_rate"] = confusion_matrix.loc["normal",:]/(performance_params["n_true"]["normal"]+1e-12) # fraction of true normals, which are predicted as malfunction
    performance_params["escape_rate"] = confusion_matrix.loc[:,"normal"]/(confusion_matrix.sum(axis=1)+1e-12) # fraction of the true malfunction that are predicted as normal
    performance_params["f1_score"] = 2 * (performance_params["efficiency"] * performance_params["purity"]) / (performance_params["efficiency"] + performance_params["purity"] + 1e-12)
    detailed_performance=performance_params.copy()
    
    performance_params_summary = {}
    if sum(performance_params["n_pred"]) == 0:
        return performance_params_summary, sp
    performance_params_summary["ri_score"] = np.average(performance_params["efficiency"], weights=performance_params["n_true"])
    expected_ri=(sum([math.comb(int(k), 2) for k in performance_params["n_true"].values]) * sum([math.comb(int(k), 2) for k in performance_params["n_pred"].values]))/math.comb(int(confusion_matrix.sum().sum()),2)**2
    performance_params_summary["ari_score"] = (performance_params_summary["ri_score"] -  expected_ri)/ (1-expected_ri)

    true_val, pred_val=[],[]
    for j in confusion_matrix.keys():
        for i in confusion_matrix.index:
            val = confusion_matrix.loc[(i,j)]
            true_val.extend(np.repeat(j, val))
            pred_val.extend(np.repeat(i, val))
    nmi = normalized_mutual_info_score(true_val, pred_val)
    ami = adjusted_mutual_info_score(true_val, pred_val)
    performance_params_summary["nmi"]=nmi
    performance_params_summary["ami"]=ami

    if not isinstance(contingency_table, type(None)):
        true_val, pred_val=[],[]
        performance_params_summary["n_clusters"]=len(contingency_table.index)
        for j in contingency_table.keys():
            for i in contingency_table.index:
                val = contingency_table.loc[(i,j)]
                true_val.extend(np.repeat(j, val))
                pred_val.extend(np.repeat(i, val))
        nmi = normalized_mutual_info_score(true_val, pred_val)
        ami = adjusted_mutual_info_score(true_val, pred_val)
        performance_params_summary["cluster_nmi"]=nmi
        performance_params_summary["cluster_ami"]=ami
        contingency_table_performance = cpm.cluster_performance_metrics(contingency_table)
        for key in contingency_table_performance.keys():
            performance_params_summary["cluster_"+key] = contingency_table_performance[key]
        del contingency_table_performance

    performance_params.drop(index=["noise", "normal"],inplace=True)
    performance_params_summary["n_noise"] = detailed_performance["n_pred"].loc["noise"]
    performance_params_summary["mean_efficiency"] = performance_params["efficiency"].mean()
    performance_params_summary["mean_purity"] = performance_params["purity"].mean()
    performance_params_summary["mean_true_malfunction_pred_noise"] = performance_params["true_malfunction_pred_noise"].mean()
    performance_params_summary["mean_true_noise_pred_malfunction"] = performance_params["true_noise_pred_malfunction"].mean()
    performance_params_summary["mean_fake_rate"] = performance_params["fake_rate"].mean()
    performance_params_summary["mean_escape_rate"] = performance_params["escape_rate"].mean()
    performance_params_summary["mean_f1_score"] = performance_params["f1_score"].mean()
    if performance_params["n_pred"].sum() > 0:
        performance_params_summary["expected_efficiency"] = np.average(performance_params["efficiency"], weights=performance_params["n_true"])
        performance_params_summary["expected_purity"] = np.average(performance_params["purity"], weights=performance_params["n_pred"])
        performance_params_summary["expected_true_malfunction_pred_noise"] = np.average(performance_params["true_malfunction_pred_noise"], weights=performance_params["n_true"])
        performance_params_summary["expected_true_noise_pred_malfunction"] = np.average(performance_params["true_noise_pred_malfunction"], weights=performance_params["n_pred"])
        performance_params_summary["expected_fake_rate"] = np.average(performance_params["fake_rate"], weights=performance_params["n_true"])
        performance_params_summary["expected_escape_rate"] = np.average(performance_params["escape_rate"], weights=performance_params["n_pred"])
        performance_params_summary["expected_f1_score"] = np.average(performance_params["f1_score"], weights=performance_params["n_true"])
    else:
        performance_params_summary["expected_efficiency"] = 0
        performance_params_summary["expected_purity"] = 0
        performance_params_summary["expected_true_malfunction_pred_noise"] = 0
        performance_params_summary["expected_true_noise_pred_malfunction"] = 0
        performance_params_summary["expected_fake_rate"] = 1
        performance_params_summary["expected_escape_rate"] = 1
        performance_params_summary["expected_f1_score"] = 0
    if not isinstance(times, type(None)):
        performance_params_summary["wall_time"] = times[0]
        performance_params_summary["cpu_time"] = times[1]
    return performance_params_summary, detailed_performance    
# predict with umap.fit and db_scan.predict
