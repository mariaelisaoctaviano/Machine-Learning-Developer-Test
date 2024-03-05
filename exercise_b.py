from helpers import extract_embeddings_and_labels, load_data, perform_kfold_validation, plot_confusion_matrix

def exercise_b():
    data = load_data('mini_gm_public_v0.1.p')
    X, y = extract_embeddings_and_labels(data)
    true_labels, predicted_labels_cosine, predicted_labels_euclidean = perform_kfold_validation(X, y)
    plot_confusion_matrix(true_labels, predicted_labels_cosine, 'Matriz de Confusão para Distância Cosseno')
    plot_confusion_matrix(true_labels, predicted_labels_euclidean, 'Matriz de Confusão para Distância Euclidiana')