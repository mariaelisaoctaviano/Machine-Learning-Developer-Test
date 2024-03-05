import unittest
from unittest.mock import patch
import exercise_c

class TestExerciseC(unittest.TestCase):
    @patch('exercise_c.save_metrics')
    @patch('exercise_c.calculate_metrics')
    @patch('exercise_c.perform_kfold_validation')
    @patch('exercise_c.extract_embeddings_and_labels')
    @patch('exercise_c.load_data')
    def test_exercise_c(self, mock_load_data, mock_extract_embeddings_and_labels, mock_perform_kfold_validation, mock_calculate_metrics, mock_save_metrics):
        # Configura os mocks para retornarem valores específicos
        mock_load_data.return_value = 'dados_mockados'
        mock_extract_embeddings_and_labels.return_value = ('X_mockado', 'y_mockado')
        mock_perform_kfold_validation.return_value = ('true_labels_mockado', 'predicted_labels_cosine_mockado', 'predicted_labels_euclidean_mockado')
        mock_calculate_metrics.side_effect = [{'metric': 'cosine_metric'}, {'metric': 'euclidean_metric'}]
        
        # Chama a função exercise_c para testar
        exercise_c.exercise_c()
        
        # Verifica se load_data foi chamado corretamente
        mock_load_data.assert_called_once_with('mini_gm_public_v0.1.p')
        
        # Verifica se extract_embeddings_and_labels foi chamado corretamente
        mock_extract_embeddings_and_labels.assert_called_once_with('dados_mockados')
        
        # Verifica se perform_kfold_validation foi chamado corretamente
        mock_perform_kfold_validation.assert_called_once_with('X_mockado', 'y_mockado')
        
        # Verifica se calculate_metrics foi chamado corretamente para ambos os conjuntos de labels
        mock_calculate_metrics.assert_any_call('true_labels_mockado', 'predicted_labels_cosine_mockado')
        mock_calculate_metrics.assert_any_call('true_labels_mockado', 'predicted_labels_euclidean_mockado')
        
        # Verifica se save_metrics foi chamado corretamente
        mock_save_metrics.assert_called_once_with({'metric': 'cosine_metric'}, {'metric': 'euclidean_metric'}, 'classification_metrics')

if __name__ == '__main__':
    unittest.main()
