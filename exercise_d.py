from helpers import binarize_labels, calculate_roc_auc, load_data, plot_comparison_roc, prepare_data

def exercise_d():
    data = load_data('mini_gm_public_v0.1.p')
    X, y = prepare_data(data)
    y_binarized = binarize_labels(y)
    micro_fpr_cos, mean_micro_tpr_cos, mean_micro_auc_cos = calculate_roc_auc(X, y_binarized, 'cosine')
    micro_fpr_euc, mean_micro_tpr_euc, mean_micro_auc_euc = calculate_roc_auc(X, y_binarized, 'euclidean')
    plot_comparison_roc(micro_fpr_cos, mean_micro_tpr_cos, mean_micro_auc_cos, micro_fpr_euc, mean_micro_tpr_euc, mean_micro_auc_euc)