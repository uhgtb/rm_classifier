"""
Algorithm to Select a Subset of Data Maximizing Label Entropy
=============================================================
Author: Johann Luca Kastner
Date: 15/09/2025
License: MIT

"""
import numpy as np
import random

def handle_few_event_classes(labels, unique_labels, remaining_n, mask):
    no_more_few_event_classes = True
    k = len(unique_labels)
    base = remaining_n // k
    for label in unique_labels:
        label_indices = labels == label
        if sum(label_indices) <= base:
            mask[label_indices] = True
            remaining_n -= sum(label_indices)
            no_more_few_event_classes = False
            unique_labels = unique_labels[unique_labels != label]
    return unique_labels, remaining_n, mask, no_more_few_event_classes

def high_entropy_subset_mask(labels, n, seed=None):
    """
    Selects a mask of size n maximizing label entropy, with all -1 labels always included.

    Args:
        labels (array-like): List or array of labels (length m).
        n (int): Desired number of selected elements.
        seed (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: Boolean mask of size m, with True for selected items.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    labels = np.array(labels)
    m = len(labels)
    unique_labels = np.unique(labels)

    if n > m:
        raise ValueError("n cannot be greater than number of datapoints")

    mask = np.zeros(m, dtype=bool)

    # Step 1: Handle all noise labels (must be included)
    noise_indices = np.where(labels == -1)[0]
    num_noise = len(noise_indices)
    if num_noise >= n:
        selected_indices = np.random.choice(noise_indices, n, replace=False)
        mask[selected_indices] = True
        return mask  # Done

    # Select all -1s
    mask[noise_indices] = True
    remaining_n = n - num_noise
    if -1 in unique_labels or "-1" in unique_labels:
        unique_labels = unique_labels[unique_labels != -1]
    
    no_more_few_event_classes = False
    while not no_more_few_event_classes: 
        unique_labels, remaining_n, mask, no_more_few_event_classes = handle_few_event_classes(labels, unique_labels, remaining_n, mask)
    if remaining_n > 0:
        k = len(unique_labels)
        base = remaining_n // k
        remainder = remaining_n % k
        for i, label in enumerate(unique_labels):
            label_indices = np.where(labels == label)[0]
            select_count = base + (1 if i < remainder else 0)
            selected_indices = np.random.choice(label_indices, select_count, replace=False)
            mask[selected_indices] = True
    return mask

