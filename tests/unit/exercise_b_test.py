import unittest
from unittest.mock import patch
import exercise_b

class TestExerciseB(unittest.TestCase):
    @patch('exercise_b.plot_confusion_matrix')
    @patch('exercise_b.perform_kfold_validation')
    @patch('exercise_b.extract_embeddings_and_labels')
    @patch('exercise_b.load_data')
    def test_exercise_b(self, mock_load_data, mock_extract_embeddings_and_labels, mock_perform_kfold_validation, mock_plot_confusion_matrix):
        # Configura os mocks para retornarem valores específicos
        mock_load_data.return_value = 'dados_mockados'
        mock_extract_embeddings_and_labels.return_value = ('X_mockado', 'y_mockado')
        mock_perform_kfold_validation.return_value = ('true_labels_mockado', 'predicted_labels_cosine_mockado', 'predicted_labels_euclidean_mockado')
        
        # Chama a função exercise_b para testar
        exercise_b.exercise_b()
        
        # Verifica se load_data foi chamado corretamente
        mock_load_data.assert_called_once_with('mini_gm_public_v0.1.p')
        
        # Verifica se extract_embeddings_and_labels foi chamado corretamente
        mock_extract_embeddings_and_labels.assert_called_once_with('dados_mockados')
        
        # Verifica se perform_kfold_validation foi chamado corretamente
        mock_perform_kfold_validation.assert_called_once_with('X_mockado', 'y_mockado')
        
        # Verifica se plot_confusion_matrix foi chamado corretamente duas vezes, uma para cada tipo de distância
        calls = [
            unittest.mock.call('true_labels_mockado', 'predicted_labels_cosine_mockado', 'Matriz de Confusão para Distância Cosseno'),
            unittest.mock.call('true_labels_mockado', 'predicted_labels_euclidean_mockado', 'Matriz de Confusão para Distância Euclidiana')
        ]
        mock_plot_confusion_matrix.assert_has_calls(calls, any_order=True)

if __name__ == '__main__':
    unittest.main()
