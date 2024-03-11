import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.calibration import label_binarize
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt, table
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pandas.plotting import table

def load_data(file_name, folder_name='data'):
    """
    Carrega dados de um arquivo pickle localizado no diretório especificado ou no diretório 'data' por padrão.
    
    Parâmetros:
    - file_name: Nome do arquivo pickle a ser carregado. Pode incluir ou não o caminho até o diretório.
    - folder_name: Nome do diretório padrão onde o arquivo está localizado, caso um caminho específico não seja fornecido.
    
    Retorna:
    - Dados carregados do arquivo pickle.
    """
    # Verifica se o caminho até o arquivo já foi fornecido
    if not os.path.dirname(file_name):
        file_path = os.path.join(folder_name, file_name)
    else:
        file_path = file_name
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        return None
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

def extract_features_and_labels(data, feature_name='embeddings', label_name='labels'):
    """
    Função auxiliar para extrair características (embeddings) e rótulos dos dados.
    
    Parâmetros:
    - data: Estrutura de dados de entrada.
    - feature_name: Nome da variável para as características extraídas.
    - label_name: Nome da variável para os rótulos extraídos.
    
    Retorna:
    - Um dicionário contendo as características e rótulos extraídos.
    """
    features = []
    labels = []
    for syndrome_id, subjects in data.items():
        for _, images in subjects.items():
            for _, encoding in images.items():
                features.append(encoding)
                labels.append(syndrome_id)
    return {feature_name: np.array(features), label_name: np.array(labels)}

def prepare_data(data):
    result = extract_features_and_labels(data, 'X', 'y')
    # Garantir que X seja um array 2D
    result['X'] = result['X'].reshape(len(result['X']), -1)
    return result['X'], result['y']

def extract_embeddings_and_labels(data):
    result = extract_features_and_labels(data)
    return result['embeddings'], result['labels']

def calculate_tsne(embeddings, n_components=2, random_state=42):
    tsne = TSNE(n_components=n_components, random_state=random_state)
    embeddings_tsne = tsne.fit_transform(embeddings)
    return embeddings_tsne

def plot_tsne(embeddings_tsne, labels):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = labels == label
        plt.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], s=10, label=label)
    plt.title('Plot dos Embeddings - t-SNE')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    plt.savefig("data/plot_tsne.png")
    plt.close()

def perform_kfold_validation(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    true_labels = []
    predicted_labels_cosine = []
    predicted_labels_euclidean = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Classify using KNN with cosine and euclidean distances
        for metric in ['cosine', 'euclidean']:
            knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            
            if metric == 'cosine':
                predicted_labels_cosine.extend(y_pred)
            else:
                predicted_labels_euclidean.extend(y_pred)
                
        true_labels.extend(y_test)
    
    return true_labels, predicted_labels_cosine, predicted_labels_euclidean

def plot_confusion_matrix(true_labels, predicted_labels, title):
    title_changed = title.replace(" ", "_").lower()
    path_file = "data/plot_confusion_matrix_" + title_changed + ".png"
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig(path_file)
    plt.close()

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

def save_metrics(metrics_dict1, metrics_dict2, file_base_name, directory='data'):
    """
    Converte dicionários de métricas em um DataFrame do pandas e salva as métricas em formatos CSV, PDF e TXT dentro do diretório especificado.
    
    Parâmetros:
    - metrics_dict1: Dicionário contendo as métricas do primeiro conjunto.
    - metrics_dict2: Dicionário contendo as métricas do segundo conjunto.
    - file_base_name: Nome base para os arquivos de saída, sem a extensão de arquivo.
    - directory: Caminho do diretório onde os arquivos serão salvos. Padrão é 'data'.
    """
    # Verificar se o diretório existe, senão cria
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Construir caminhos completos dos arquivos
    csv_file = os.path.join(directory, f'{file_base_name}.csv')
    txt_file = os.path.join(directory, f'{file_base_name}.txt')
    pdf_file = os.path.join(directory, f'{file_base_name}.pdf')

    # Converter dicionários em DataFrame
    metrics_df = pd.DataFrame({'Metric': list(metrics_dict1.keys()),
                               'Cosine': list(metrics_dict1.values()),
                               'Euclidean': list(metrics_dict2.values())}).set_index('Metric')
    
    # Salvar como CSV
    metrics_df.to_csv(csv_file, index=True)
    print(f"Métricas salvas em: {csv_file}")
    
    # Salvar como TXT
    with open(txt_file, 'w') as f:
        f.write(metrics_df.reset_index().to_string(index=False))
    print(f"Métricas salvas em: {txt_file}")
    
    # Salvar como PDF - Utilizando matplotlib para criar uma tabela
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    tbl = table(ax, metrics_df, loc='center', cellLoc='center', colWidths=[0.2]*len(metrics_df.columns)+[0.1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    print(f"Métricas salvas em: {pdf_file}")

def binarize_labels(y):
    return label_binarize(y, classes=np.unique(y))

def calculate_roc_auc(X, y_binarized, metric='cosine'):
    n_classes = y_binarized.shape[1]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    micro_fpr = np.linspace(0, 1, 100)
    micro_tprs = []

    for i in range(n_classes):
        for train_index, test_index in kf.split(X, y_binarized):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_binarized[train_index], y_binarized[test_index]
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric=metric)
            knn.fit(X_train, y_train[:, i])
            y_score = knn.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test[:, i], y_score)
            micro_tprs.append(np.interp(micro_fpr, fpr, tpr))
            micro_tprs[-1][0] = 0.0

    mean_micro_tpr = np.mean(micro_tprs, axis=0)
    mean_micro_tpr[-1] = 1.0
    mean_micro_auc = auc(micro_fpr, mean_micro_tpr)
    
    return micro_fpr, mean_micro_tpr, mean_micro_auc

def plot_comparison_roc(micro_fpr_cos, mean_micro_tpr_cos, mean_micro_auc_cos,
                        micro_fpr_euc, mean_micro_tpr_euc, mean_micro_auc_euc):
    plt.plot(micro_fpr_cos, mean_micro_tpr_cos, color='blue', label='Micro-average ROC: Cosine (area = {:.2f})'.format(mean_micro_auc_cos))
    plt.plot(micro_fpr_euc, mean_micro_tpr_euc, color='green', label='Micro-average ROC: Euclidean (area = {:.2f})'.format(mean_micro_auc_euc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison - Cosine vs Euclidean Distances')
    plt.legend(loc="lower right")
    plt.savefig("data/plot_comparison_roc.png")
    plt.close()

