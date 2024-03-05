import unittest
from unittest.mock import patch
import exercise_a

class TestExerciseA(unittest.TestCase):
    @patch('exercise_a.plot_tsne')
    @patch('exercise_a.calculate_tsne')
    @patch('exercise_a.extract_embeddings_and_labels')
    @patch('exercise_a.load_data')
    def test_exercise_a(self, mock_load_data, mock_extract_embeddings_and_labels, mock_calculate_tsne, mock_plot_tsne):
        # Configura os mocks para retornarem valores específicos
        mock_load_data.return_value = 'dados_mockados'
        mock_extract_embeddings_and_labels.return_value = ('embeddings_mockados', 'labels_mockados')
        mock_calculate_tsne.return_value = 'embeddings_tsne_mockados'
        
        # Chama a função exercise_a para testar
        exercise_a.exercise_a()
        
        # Verifica se load_data foi chamado corretamente
        mock_load_data.assert_called_once_with('mini_gm_public_v0.1.p')
        
        # Verifica se extract_embeddings_and_labels foi chamado corretamente
        mock_extract_embeddings_and_labels.assert_called_once_with('dados_mockados')
        
        # Verifica se calculate_tsne foi chamado corretamente
        mock_calculate_tsne.assert_called_once_with('embeddings_mockados')
        
        # Verifica se plot_tsne foi chamado corretamente
        mock_plot_tsne.assert_called_once_with('embeddings_tsne_mockados', 'labels_mockados')

if __name__ == '__main__':
    unittest.main()
