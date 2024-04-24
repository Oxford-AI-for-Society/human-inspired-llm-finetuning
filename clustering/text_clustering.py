import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import logging
import umap.umap_ as umap  
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_embeddings(prompts, embedding_model='all-mpnet-base-v2', batch_size=32, cache_dir='./cache'):
    """
    Compute the text embeddings using pr≠=etrained embedding mdoels
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model = SentenceTransformer(embedding_model, cache_folder=cache_dir).to(device)
    embeddings = model.encode(prompts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    return embeddings.cpu().numpy()


def generate_clusters(embeddings, n_neighbors, n_components, min_cluster_size, min_samples=None, random_state=42):
    """
    Generate clusters by using umap for dimension reduction and hdbscan for clustering
    """
    umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                min_dist=0.0, 
                                metric='cosine',
                                random_state=random_state).fit_transform(embeddings)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                min_samples=min_samples, 
                                metric='euclidean', 
                                cluster_selection_method='eom')
    clusterer.fit(umap_embeddings)
    
    return clusterer


def score_clusters(clusters, prob_threshold=0.05):
    """
    Define the score function as the percent of data with < 5% cluster label confidence, which we will minimise to reduce noise
    The probabilities attribute of hdbscan measures the strength with which each sample is a member of its assigned cluster
    """
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(cluster_labels)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)
    return label_count, cost


def objective(params, embeddings, label_lower, label_upper):
    """
    Compute the objective function with additional penalty on too few or too many cluster numbers
    """
    clusters = generate_clusters(embeddings, 
                                 n_neighbors=int(params['n_neighbors']), 
                                 n_components=int(params['n_components']), 
                                 min_cluster_size=int(params['min_cluster_size']),
                                 random_state=int(params['random_state']))
    
    label_count, cost = score_clusters(clusters, prob_threshold=0.05)

    # Add 15% penalty on the cost function if outside the desired range of clusters
    if (label_count < label_lower) or (label_count > label_upper):
        penalty = 0.20
    else:
        penalty = 0
    
    return {'loss': cost + penalty, 'label_count': label_count, 'status': STATUS_OK}


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayseian search on hyperopt hyperparameter space to minimize objective function
    """
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    best = fmin(fmin_objective, 
                space=space, 
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=trials)
    
    best_params = space_eval(space, best)
    print('best:')
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")
    
    best_clusters = generate_clusters(embeddings, 
                                      n_neighbors=int(best_params['n_neighbors']), 
                                      n_components=int(best_params['n_components']), 
                                      min_cluster_size=int(best_params['min_cluster_size']),
                                      random_state=int(best_params['random_state']))
    
    return best_params, best_clusters, trials


# def compute_tfidf_and_top_words(texts, cluster_labels, n_top=10):
#     """
#     Calculate TF-IDF and extract top words (unigrams and bigrams) for each cluster
#     """
#     df = pd.DataFrame({'text': texts, 'cluster': cluster_labels})
#     results = {}
    
#     for cluster in sorted(df['cluster'].unique()):
#         cluster_texts = df[df['cluster'] == cluster]['text'].values
#         if len(cluster_texts) > 1:  # Ensure there are enough texts for TF-IDF
#             vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
#             tfidf_matrix = vectorizer.fit_transform(cluster_texts)
#             feature_array = np.array(vectorizer.get_feature_names_out())
#             tfidf_sorting = np.argsort(tfidf_matrix.sum(axis=0)).flatten()[::-1]
            
#             top_n = feature_array[tfidf_sorting][:n_top]
#             results[cluster] = top_n
#         else:
#             results[cluster] = []

#     return results


# def visualize_clusters_with_annotations(umap_embeddings, cluster_labels, top_words):
#     """
#     Visualise clusters in 2D UMAP dimensions
#     """
#     plt.figure(figsize=(12, 10))
#     scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("Clusters Visualized with UMAP")
#     plt.xlabel("UMAP Dimension 1")
#     plt.ylabel("UMAP Dimension 2")
    
#     # # Adding text annotations (top words) for each cluster
#     # for cluster in unique_labels:
#     #     # Find the centroid of each cluster to place the annotation
#     #     centroid = umap_embeddings[cluster_labels == cluster].mean(axis=0)
#     #     top_word = top_words[cluster][0] if len(top_words[cluster]) > 0 else ''
#     #     annotation = f"Cluster {cluster}\n{top_word}"
#     #     plt.annotate(annotation, (centroid[0], centroid[1]), fontsize=9, ha='center')
    
#     plt.show()


def visualize_clusters(embeddings, cluster_labels, model_name, n_neighbors, n_components):
    """
    Visualize clusters using UMAP reduction to 2D with the best hyperparameters for each embedding model
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_labels, cmap='Spectral', s=5)
    plt.colorbar()
    plt.title(f"Clusters Visualized with UMAP for {model_name} using best parameters")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()


def main(file_path):
    # Load data with a 'prompt' column containing the text data
    data = pd.read_csv(file_path)

    if 'prompt' not in data.columns:
        logging.error("CSV file does not contain a 'prompt' column.")
        return
    
    prompts = data['prompt'].dropna().unique()
    
    # List of embedding models to evaluate
    embedding_models = [
        'pritamdeka/S-PubMedBert-MS-MARCO'
    ]
    
    # Define the search space for hyperparameters
    space = {
    'n_neighbors': hp.choice('n_neighbors', range(5, 20)), # number of neighbours to consider when doing UMAP
    'n_components': hp.choice('n_components', range(3, 15)), # dimensions of UMAP
    'min_cluster_size': hp.choice('min_cluster_size', range(25, 35)), # the minimum number of data points to be considered a cluster
    'random_state': 42
}
    
    results = {}
    
    for model in embedding_models:
        print(f"Processing model: {model}")
        
        # Compute embeddings for the current model
        embeddings = compute_embeddings(prompts, embedding_model=model)
        
        # Run Bayesian optimization
        label_lower = 6
        label_upper = 20
        best_params, best_clusters, trials = bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100)
        
        # Visualize the clusters using the best hyperparameters
        visualize_clusters(embeddings, best_clusters.labels_, model_name=model, n_neighbors=int(best_params['n_neighbors']), n_components=2)
        
        # Store results for each model
        results[model] = {
            'best_params': best_params,
            'best_clusters': best_clusters,
            'trials': trials
        }
        
    # Return or print all results
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process a 'prompt' column in a CSV file including clustering and visualization.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    all_results = main(args.file_path)
    # print(all_results) 
