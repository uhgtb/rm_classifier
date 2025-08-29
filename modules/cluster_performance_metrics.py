import numpy as np
import pandas as pd

def get_main_malfunction(contingency_table):
    """Get the malfunction that is most represented in each cluster
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: malfunction that is most represented in each cluster
    """
    return contingency_table.idxmax(axis=1)

def get_second_malfunction(contingency_table):
    """Get the malfunction that is second most represented in each cluster
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: malfunction that is second most represented in each cluster
    """
    temp = contingency_table.copy()
    for idx in temp.index:
        max_col = temp.idxmax(axis=1)[idx]
        temp.loc[idx, max_col] = np.nan
    return temp.idxmax(axis=1)

def get_main_cluster(contingency_table):
    """Get the cluster that is most represented for each malfunction
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: cluster that is most represented for each malfunction
    """
    return contingency_table.idxmax(axis=0)

def get_malfunction_separability(contingency_table):
    """Get the separability of each malfunction
    separability = sum_i (n_i^2 / N_i) / M
    where n_i is the number of events from the malfunction in cluster i, N_i is the total number of events in cluster i, and M is the total number of events from the malfunction
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: separability of each malfunction
    """
    cluster_sizes = contingency_table.sum(axis=1)
    malfunction_sizes = contingency_table.sum(axis=0)
    separability = pd.Series(index=contingency_table.columns)
    for malfunction in contingency_table.columns:
        # Get the cluster counts for the current malfunction
        separability[malfunction] = np.sum([contingency_table[malfunction][cluster]**2/cluster_sizes[cluster] for cluster in contingency_table.index if cluster!="noise"])/malfunction_sizes[malfunction]
    return separability

def get_cluster_purity(contingency_table):
    """Get the purity of each cluster
    purity = sum_j (n_j^2 / N) / N
    where n_j is the number of events from malfunction j in the cluster, N is the total number of events in the cluster
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: purity of each cluster
    """
    cluster_sizes = contingency_table.sum(axis=1)
    purity = pd.Series(index=contingency_table.index)
    for cluster in contingency_table.index:
        try:
            purity[cluster] = np.sum([contingency_table[malfunction][cluster]**2/cluster_sizes[cluster] for malfunction in contingency_table.columns])/cluster_sizes[cluster]
        except:
            pass
    return purity

def get_cluster_fraction_of_dominant_malfunction(contingency_table):
    """Get the fraction of the dominant malfunction in each cluster
    Args: 
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: fraction of the dominant malfunction in each cluster 
    """
    cluster_sizes = contingency_table.sum(axis=1)
    cluster_fraction = pd.Series(index=contingency_table.index)
    dominant_malfunctions = contingency_table.idxmax(axis=1)
    for cluster in contingency_table.index:
        # Get the cluster counts for the current malfunction
        try:
            cluster_fraction[cluster] = contingency_table[dominant_malfunctions[cluster]][cluster]/cluster_sizes[cluster]
        except: 
            pass
    return cluster_fraction

def get_second_malfunction(contingency_table):
    """Get the malfunction that is second most represented in each cluster
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: malfunction that is second most represented in each cluster
    """
    temp = contingency_table.copy()
    for idx in temp.index:
        max_col = temp.idxmax(axis=1)[idx]
        temp.loc[idx, max_col] = np.nan
    return temp.idxmax(axis=1)

def get_third_malfunction(contingency_table):
    """Get the malfunction that is third most represented in each cluster
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: malfunction that is third most represented in each cluster
    """
    temp = contingency_table.copy()
    for idx in temp.index:
        # Remove first max
        max_col = temp.idxmax(axis=1)[idx]
        temp.loc[idx, max_col] = np.nan
        # Remove second max
        max_col = temp.idxmax(axis=1)[idx]
        temp.loc[idx, max_col] = np.nan
    return temp.idxmax(axis=1)

def get_cluster_fraction_of_second_dominant_malfunction(contingency_table):
    """Get the fraction of the second dominant malfunction in each cluster
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: fraction of the second dominant malfunction in each cluster
    """
    malfunction2 = get_second_malfunction(contingency_table)
    cluster_sizes = contingency_table.sum(axis=1)
    cluster_fraction = pd.Series(index=contingency_table.index)
    for cluster in contingency_table.index:
        # Get the cluster counts for the current malfunction
        try:
            cluster_fraction[cluster] = contingency_table[malfunction2[cluster]][cluster]/cluster_sizes[cluster]
        except: 
            pass
    return cluster_fraction

def get_cluster_fraction_of_third_dominant_malfunction(contingency_table):
    """Get the fraction of the third dominant malfunction in each cluster
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: fraction of the third dominant malfunction in each cluster
    """
    malfunction3 = get_third_malfunction(contingency_table)
    cluster_sizes = contingency_table.sum(axis=1)
    cluster_fraction = pd.Series(index=contingency_table.index)
    for cluster in contingency_table.index:
        # Get the cluster counts for the current malfunction
        try:
            cluster_fraction[cluster] = contingency_table[malfunction3[cluster]][cluster]/cluster_sizes[cluster]
        except: 
            pass
    return cluster_fraction

def get_malfunction_fraction_of_dominated_clusters(contingency_table):
    """Fraction of the malfunction in the clusters which are dominated by the malfunction
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: fraction of the malfunction in the clusters which are dominated by this malfunction
    """
    cluster_sizes = contingency_table.sum(axis=1)
    malfunction_sizes = contingency_table.sum(axis=0)
    fidc = pd.Series(index=contingency_table.columns)
    mm=contingency_table.idxmax(axis=1)
    for malfunction in contingency_table.columns:
        clusters=mm[mm==malfunction]
        fidc[malfunction] = np.sum([contingency_table[malfunction][cluster]**2/cluster_sizes[cluster] for cluster in clusters.index])/malfunction_sizes[malfunction]
    return fidc

def get_malfunction_accuracy(contingency_table):
    """Accuracy/Efficiency, i.e. the fraction of the events from a malfunction which are mapped into a cluster which is dominated by the malfunction
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: accuracy of each malfunction
    """
    malfunction_sizes = contingency_table.sum(axis=0)
    accuracy = pd.Series(index=contingency_table.columns)
    mm=contingency_table.idxmax(axis=1)
    for malfunction in contingency_table.columns:
        clusters=mm[mm==malfunction]
        accuracy[malfunction] = np.sum([contingency_table[malfunction][cluster] for cluster in clusters.index])/malfunction_sizes[malfunction]
    return accuracy

def get_malfunction_number_of_relevant_clusters(contingency_table, threshold=100):
    """Fraction of the events from a malfunction which are mapped into a cluster which is dominated by the malfunction
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
        threshold (int): minimum number of events from the malfunction in the cluster to be considered relevant
    Returns:
        pd.Series: number of relevant clusters for each malfunction
    """
    mnorc = pd.Series(index=contingency_table.columns)
    for malfunction in contingency_table.columns:
        mnorc[malfunction] = int(np.sum(contingency_table[malfunction]>threshold))
    return mnorc

def get_malfunction_classified_normal(contingency_table):
    """Get the fraction of events from a malfunction that are classified as normal
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: fraction of events from a malfunction that are classified as normal
    """
    malfunction_sizes = contingency_table.sum(axis=0)
    mcn = pd.Series(index=contingency_table.columns)
    mm=contingency_table.idxmax(axis=1)
    for malfunction in contingency_table.columns:
        clusters=mm[mm=="normal"]
        mcn[malfunction] = np.sum([contingency_table[malfunction][cluster] for cluster in clusters.index])/malfunction_sizes[malfunction]
    return mcn

def get_normal_classified_as_malfunction(contingency_table):
    """Get the fraction of events from normal that are classified as malfunction
    Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.Series: fraction of events from normal that are classified as malfunction
    """
    malfunction_sizes = contingency_table.sum(axis=0)
    mcn = pd.Series(index=contingency_table.columns)
    mm=contingency_table.idxmax(axis=1)
    for malfunction in contingency_table.columns:
        clusters=mm[mm==malfunction]
        mcn[malfunction] = np.sum([contingency_table["normal"][cluster] for cluster in clusters.index])/malfunction_sizes["normal"]
    return mcn

def malfunction_characteristics(contingency_table):
    """Get main characteristics of each malfunction
     Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.DataFrame: characteristics of each malfunction
    """
    dict={}
    dict["malfunction_size"] = contingency_table.sum(axis=0).astype(int)
    dict["main_cluster"] = contingency_table.idxmax(axis=0)
    dict["separability"] = get_malfunction_separability(contingency_table)
    dict["accuracy"] = get_malfunction_accuracy(contingency_table)
    dict["fraction_of_dominated_clusters"] = get_malfunction_fraction_of_dominated_clusters(contingency_table)
    dict["malfunction_classified_as_normal"] = get_malfunction_classified_normal(contingency_table)
    dict["normal_classified_as_malfunction"] = get_normal_classified_as_malfunction(contingency_table)
    dict["n_dominated_clusters"] = contingency_table.idxmax(axis=1).value_counts().astype(int)
    return pd.DataFrame(dict)

def cluster_characteristics(contingency_table):
    """Get main characteristics of each cluster
     Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        pd.DataFrame: characteristics of each cluster
    """
    dict={}
    dict["cluster_size"] = contingency_table.sum(axis=1).astype(int)
    dict["purity"] = get_cluster_purity(contingency_table)
    dict["main_malfunction"] = contingency_table.idxmax(axis=1)
    dict["fraction_of_dominant_malfunction"] = get_cluster_fraction_of_dominant_malfunction(contingency_table)
    dict["second_malfunction"] = get_second_malfunction(contingency_table)
    dict["fraction_of_second_dominant_malfunction"] = get_cluster_fraction_of_second_dominant_malfunction(contingency_table)
    dict["third_malfunction"] = get_third_malfunction(contingency_table)
    dict["fraction_of_third_dominant_malfunction"] = get_cluster_fraction_of_third_dominant_malfunction(contingency_table)

    return pd.DataFrame(dict)

def cluster_performance_metrics(contingency_table):
    """Get overall performance metrics of the clustering
     Args:
        contingency_table (pd.DataFrame): contingency table with clusters as index and malfunctions as columns
    Returns:
        dict: overall performance metrics of the clustering
    """
    cc=cluster_characteristics(contingency_table)
    clean_cc=cc[cc.index != "noise"]
    mc=malfunction_characteristics(contingency_table)
    abnormal_mc = mc[(mc.index != "normal") & (mc.index != "noise")]
    dict={}
    dict["n_clusters"]= len(cc)
    try:
        dict["n_noise"] = cc["cluster_size"]["noise"]
    except:
        dict["n_noise"] = 0
    if abnormal_mc["malfunction_size"].sum() > 0:
        dict["expected_separability"] = np.average(abnormal_mc['separability'], weights=list(abnormal_mc["malfunction_size"]))
        dict["expected_accuracy"] = np.average(abnormal_mc['accuracy'], weights=list(abnormal_mc["malfunction_size"]))
        dict["expected_malfunction_classified_as_normal"] = np.average(abnormal_mc['malfunction_classified_as_normal'], weights=list(abnormal_mc["malfunction_size"]))
    else:
        dict["expected_separability"] = 0
        dict["expected_accuracy"] = 0
        dict["expected_malfunction_classified_as_normal"] = 0
    if clean_cc["cluster_size"].sum() > 0:
        dict["expected_purity"] = np.average(clean_cc['purity'], weights=list(clean_cc["cluster_size"]))
    else:
        dict["expected_purity"] = 0 
    dict["mean_separability"]= abnormal_mc["separability"].mean()
    dict["mean_accuracy"]= abnormal_mc["accuracy"].mean()
    dict["mean_malfunction_classified_as_normal"]= abnormal_mc["malfunction_classified_as_normal"].mean()
    dict["mean_purity"]= clean_cc["purity"].mean()
    dict["false_alerts"] = abnormal_mc["normal_classified_as_malfunction"].sum()
    return dict