import os
from unittest import mock
from helpers import load_data 

base_path = os.path.dirname(__file__)

@mock.patch("builtins.open", mock.mock_open())
@mock.patch("os.path.dirname")
@mock.patch("os.path.join")
@mock.patch("pickle.load")
def test_load_data_success_default_folder(mock_load, mock_join, mock_dirname):
    # Preparação
    file_name = 'test_data.pkl'
    test_data = {'a': 1}
    mock_dirname.return_value = False
    mock_join.return_value = os.path.join('data', file_name)
    mock_load.return_value = test_data

    # Teste
    result = load_data(file_name)
    
    # Verificação
    assert result == test_data
    mock_dirname.assert_called_with(file_name)
    mock_join.assert_called_with('data', file_name)
    mock_load.assert_called_once()

@mock.patch("builtins.open", mock.mock_open())
@mock.patch("os.path.dirname")
@mock.patch("os.path.join")
@mock.patch("pickle.load")
def test_load_data_success_specified_folder(mock_load, mock_join, mock_dirname):
    # Preparação
    file_name = 'test_data.pkl'
    test_data = {'b': 2}
    folder_name = 'temp_data'
    mock_dirname.return_value = False
    mock_join.return_value = os.path.join(folder_name, file_name)
    mock_load.return_value = test_data

    # Teste
    result = load_data(file_name, folder_name)
    
    # Verificação
    assert result == test_data
    mock_dirname.assert_called_with(file_name)
    mock_join.assert_called_with(folder_name, file_name)
    mock_load.assert_called_once()

@mock.patch("builtins.open", mock.mock_open(), side_effect=FileNotFoundError)
def test_load_data_file_not_found(mock_open):
    # Teste
    result = load_data('nonexistent_file.pkl')
    
    # Verificação
    assert result is None
    mock_open.assert_called_once()

@mock.patch("builtins.open", mock.mock_open(), side_effect=Exception("Simulated Error"))
def test_load_data_unexpected_error(mock_open):
    # Teste
    result = load_data('test_data.pkl')
    
    # Verificação
    assert result is None
    mock_open.assert_called_once()
