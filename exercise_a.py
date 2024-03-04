from helpers import calculate_tsne, extract_embeddings_and_labels, load_data, plot_tsne

def exercise_a():
    data = load_data('mini_gm_public_v0.1.p')
    embeddings, labels = extract_embeddings_and_labels(data)
    embeddings_tsne = calculate_tsne(embeddings)
    plot_tsne(embeddings_tsne, labels)