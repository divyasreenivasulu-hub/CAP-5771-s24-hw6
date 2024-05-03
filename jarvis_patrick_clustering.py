import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
from numpy.typing import NDArray
import pickle
import matplotlib.backends.backend_pdf as pdf

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def plot_clustering(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def compute_SSE(data, labels):
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse

def adjusted_rand_index(true_labels, pred_labels):
    contingency_matrix = np.zeros((np.max(true_labels) + 1, np.max(pred_labels) + 1), dtype=np.int64)
    for i in range(len(true_labels)):
        contingency_matrix[true_labels[i], pred_labels[i]] += 1

    k = np.sum(contingency_matrix, axis=1)
    l = np.sum(contingency_matrix, axis=0)
    n = np.sum(contingency_matrix)
    kl_sum = np.sum(k * (k - 1)) / 2
    ab_sum = np.sum(l * (l - 1)) / 2
    k_sum_2 = np.sum(k * (k - 1)) / 2
    l_sum_2 = np.sum(l * (l - 1)) / 2

    ab = np.sum(contingency_matrix * (contingency_matrix - 1)) / 2

    expected_index = k_sum_2 * l_sum_2 / n / (n - 1) + ab ** 2 / n / (n - 1)
    max_index = (k_sum_2 + l_sum_2) / 2
    return (ab - expected_index) / (max_index - expected_index)

def jarvis_patrick(data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    k = params_dict['k']  # Number of neighbors to consider
    s_min = params_dict['s_min']   # Similarity threshold
    
    n = len(data)
    computed_labels = np.zeros(n, dtype=np.int32)

    for i in range(n):
        # Compute distances from the current data point to all other points
        distances = cdist([data[i]], data, metric='euclidean')[0]

        # Sort distances and get the indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[1:k+1]  # Exclude the current point itself

        # Count labels of nearest neighbors
        label_counts = np.bincount(labels[nearest_indices])

        # Get the label with the highest count
        majority_label = np.argmax(label_counts)

        # Compute similarity between the current point and its nearest neighbor
        similarity = label_counts[majority_label] / k

        # Assign cluster label if similarity meets the threshold
        if similarity >= s_min:
            computed_labels[i] = majority_label + 1  # Add 1 to avoid 0 as a label
  
    # Calculate ARI
    ARI = adjusted_rand_index(labels, computed_labels)

    # Calculate SSE manually
    SSE = compute_SSE(data, computed_labels)

    return computed_labels, SSE, ARI

def hyperparameter_study(data, labels, k_range, s_min_range, num_trials):
    best_ARI = -1
    best_k = None
    best_s_min = None

    for k in k_range:
        for s_min in s_min_range:
            total_ARI = 0
            for _ in range(num_trials):
                params_dict = {'k': k, 's_min': s_min}
                computed_labels, ARI, _ = jarvis_patrick(data, labels, params_dict)
                total_ARI += ARI

            avg_ARI = total_ARI / num_trials
            if avg_ARI > best_ARI:
                best_ARI = avg_ARI
                best_k = k
                best_s_min = s_min

    return best_k, best_s_min

def compute_scores(data, labels, k_values, s_min_values):
    sse_scores = np.zeros((len(s_min_values), len(k_values)))
    ari_scores = np.zeros((len(s_min_values), len(k_values)))
    for i, k in enumerate(k_values):
        params_dict = {'k': k, 's_min': s_min_values[i]}  # Corrected 'k' and 's_min' values
        _, sse, ari = jarvis_patrick(data, labels, params_dict)
        sse_scores[i, :] = sse  # Corrected indexing
        ari_scores[i, :] = ari  # Corrected indexing
    return sse_scores, ari_scores


def jarvis_patrick_clustering():
    answers = {}

    answers["jarvis_patrick_function"] = jarvis_patrick

    cluster_data = np.load('question1_cluster_data.npy')
    cluster_labels = np.load('question1_cluster_labels.npy')

    random_indices = np.random.choice(len(cluster_data), size=5000, replace=False)
    data_subset = cluster_data[random_indices]
    labels_subset = cluster_labels[random_indices]

    data_subset = data_subset[:1000]
    labels_subset = labels_subset[:1000]

    k_range = [3, 4, 5, 6, 7, 8]
    s_min_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    num_trials = 10

    best_k, best_s_min = hyperparameter_study(data_subset, labels_subset, k_range, s_min_range, num_trials)

    params_dict = {'k': best_k, 's_min': best_s_min}

    groups = {}
    plots_values = {}

    for i in [0, 1, 2, 3, 4]:
        data_slice = cluster_data[i * 1000: (i + 1) * 1000]
        labels_slice = cluster_labels[i * 1000: (i + 1) * 1000]
        computed_labels, sse, ari = jarvis_patrick(data_slice, labels_slice, params_dict)
        groups[i] = {"smin": best_s_min, "k": best_k, "ARI": ari, "SSE": sse}
        plots_values[i] = {"computed_labels": computed_labels, "ARI": ari, "SSE": sse}

    highest_ari = -1
    best_dataset_index = None
    for i, group_info in plots_values.items():
        if group_info['ARI'] > highest_ari:
            highest_ari = group_info['ARI']
            best_dataset_index = i

    pdf_pages = pdf.PdfPages("jarvis_patrick_clustering_plots.pdf")

    plt.figure(figsize=(8, 6))
    plot_ARI = plt.scatter(cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 0],
                           cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 1],
                           c=plots_values[best_dataset_index]["computed_labels"], cmap='viridis')
    plt.title(
        f'Clustering for Dataset {best_dataset_index} (Highest ARI) with k value :{best_k} and s_min: {best_s_min}')
    plt.suptitle('Jarvis - Patrick Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Clustering')
    plt.grid(True)
    pdf_pages.savefig()
    plt.close()

    k_values = np.array(k_range)
    s_min_values = np.array(s_min_range)
    sse_scores, ari_scores = compute_scores(data_subset, labels_subset, k_values, s_min_values)

    plt.figure(figsize=(8, 6))
    plt.scatter(np.tile(k_values, len(s_min_range)), np.repeat(s_min_values, len(k_range)), c =sse_scores.flatten(),
                        cmap='viridis', s=25)
    plt.title("sse with diff k_values and s_min")
    plt.xlabel('k')
    plt.ylabel('smin')
    plt.colorbar(label='clustering')
    plt.grid(True)
    pdf_pages.savefig()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(np.tile(k_values, len(s_min_range)), np.repeat(s_min_values, len(k_range)), c =ari_scores.flatten(),
                        cmap='viridis', s=25)
    plt.title("ari with diff k_values and s_min")
    plt.xlabel('k')
    plt.ylabel('smin')
    plt.colorbar(label='clustering')
    plt.grid(True)
    pdf_pages.savefig()
    plt.close()

    lowest_sse = float('inf')
    best_dataset_index_sse = None
    for i, group_info in plots_values.items():
        if group_info['SSE'] < lowest_sse:
            lowest_sse = group_info['SSE']
            best_dataset_index_sse = i

    plt.figure(figsize=(8, 6))
    plot_SSE = plt.scatter(cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 0],
                           cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 1],
                           c=plots_values[best_dataset_index_sse]["computed_labels"], cmap='viridis')
    plt.title(
        f'Clustering for Dataset {best_dataset_index_sse} (Lowest SSE) with k value :{best_k} and s_min: {best_s_min}')
    plt.suptitle('Jarvis - Patrick Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Clustering')
    plt.grid(True)
    pdf_pages.savefig()
    plt.close()

    pdf_pages.close()

    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"]

    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    ari_values = [group_info["ARI"] for group_info in groups.values()]
    mean_ari = np.mean(ari_values)
    std_ari = np.std(ari_values)

    answers["mean_ARIs"] = mean_ari
    answers["std_ARIs"] = std_ari

    sse_values = [group_info["SSE"] for group_info in groups.values()]
    mean_sse = np.mean(sse_values)
    std_sse = np.std(sse_values)

    answers["mean_SSEs"] = mean_sse
    answers["std_SSEs"] = std_sse

    return answers

if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
