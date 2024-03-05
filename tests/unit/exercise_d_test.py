import unittest
from unittest.mock import patch
import exercise_d

class TestExerciseD(unittest.TestCase):
    @patch('exercise_d.plot_comparison_roc')
    @patch('exercise_d.calculate_roc_auc')
    @patch('exercise_d.binarize_labels')
    @patch('exercise_d.prepare_data')
    @patch('exercise_d.load_data')
    def test_exercise_d(self, mock_load_data, mock_prepare_data, mock_binarize_labels, mock_calculate_roc_auc, mock_plot_comparison_roc):
        # Configura os mocks para retornarem valores específicos
        mock_load_data.return_value = 'dados_mockados'
        mock_prepare_data.return_value = ('X_mockado', 'y_mockado')
        mock_binarize_labels.return_value = 'y_binarizado_mockado'
        mock_calculate_roc_auc.side_effect = [
            ('micro_fpr_cos_mock', 'mean_micro_tpr_cos_mock', 'mean_micro_auc_cos_mock'),
            ('micro_fpr_euc_mock', 'mean_micro_tpr_euc_mock', 'mean_micro_auc_euc_mock')
        ]
        
        # Chama a função exercise_d para testar
        exercise_d.exercise_d()
        
        # Verifica se load_data foi chamado corretamente
        mock_load_data.assert_called_once_with('mini_gm_public_v0.1.p')
        
        # Verifica se prepare_data foi chamado corretamente
        mock_prepare_data.assert_called_once_with('dados_mockados')
        
        # Verifica se binarize_labels foi chamado corretamente
        mock_binarize_labels.assert_called_once_with('y_mockado')
        
        # Verifica se calculate_roc_auc foi chamado corretamente para ambos os métodos
        mock_calculate_roc_auc.assert_any_call('X_mockado', 'y_binarizado_mockado', 'cosine')
        mock_calculate_roc_auc.assert_any_call('X_mockado', 'y_binarizado_mockado', 'euclidean')
        
        # Verifica se plot_comparison_roc foi chamado corretamente
        mock_plot_comparison_roc.assert_called_once_with(
            'micro_fpr_cos_mock', 'mean_micro_tpr_cos_mock', 'mean_micro_auc_cos_mock',
            'micro_fpr_euc_mock', 'mean_micro_tpr_euc_mock', 'mean_micro_auc_euc_mock'
        )

if __name__ == '__main__':
    unittest.main()
