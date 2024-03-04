from helpers import calculate_metrics, extract_embeddings_and_labels, load_data, perform_kfold_validation, save_metrics

def exercise_c():
    data = load_data('mini_gm_public_v0.1.p')
    X, y = extract_embeddings_and_labels(data)
    true_labels, predicted_labels_cosine, predicted_labels_euclidean = perform_kfold_validation(X, y)
    metrics_cosine = calculate_metrics(true_labels, predicted_labels_cosine)
    metrics_euclidean = calculate_metrics(true_labels, predicted_labels_euclidean)
    save_metrics(metrics_cosine, metrics_euclidean, 'classification_metrics')
