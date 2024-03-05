# Machine-Learning-Developer-Test
Bem vindo a este repositório! Aqui você vai encontrar códigos de referentes ao Teste de Desenvolvedor de Aprendizado de Máquina.

# Estrutura do Repositório
- Tests: Nesta pasta há a implementação de um teste unitário
- Images: Contém as imagens mostradas no report
- helper.py: Contém as funções compartilhadas entre os exercícios
- exercícios_n.py: Contém a solução dos exercícos

# Figuras
- Abaixo são mostradas evidências das implementações realizadas
<p align="center">
  <b>Confusion Matrix - Cosine Distance</b><br>
  <img src="https://github.com/mariaelisaoctaviano/Machine-Learning-Developer-Test/blob/main/images/confusion_matrix_cos.png" width="500" height="410"/>
</p>
<p align="center">
  <b>Confusion Matrix - Euclidean Distance</b><br>
  <img src="https://github.com/mariaelisaoctaviano/Machine-Learning-Developer-Test/blob/main/images/confusion_matrix_euclidean.png" width="500" height="410"/>
</p>

<p align="center">
  <b>ROC Plot</b><br>
  <img src="https://github.com/mariaelisaoctaviano/Machine-Learning-Developer-Test/blob/main/images/roc_plot.png" width="500" height="375"/>
</p>
<p align="center">
  <b>t-SNE Plot</b><br>
  <img src="https://github.com/mariaelisaoctaviano/Machine-Learning-Developer-Test/blob/main/images/tsne_plot.png" width="500" height="375"/>
</p>

<p align="center">
  <b>Performance Metrics</b><br>
  <img src="https://github.com/mariaelisaoctaviano/Machine-Learning-Developer-Test/blob/main/images/metrics.png"/>
</p>

# Passos para executar o código

## Sem o docker
1. Clone o repositório
2. Instale as dependências
```bash
pip install -r requirements.txt
```
3. Execute o arquivo main.py
```bash
python main.py
4. Executando os testes
```bash
python -m unittest discover -s tests -p "*_test.py"
```

## Com o docker
1. Clone o repositório
2. Crie a imagem
```bash
docker build -t ml-dev-test .
```
3. Execute o container
```bash
docker run -v ./data:/app/data ml-dev-test
```
Obs:
  - O volume é montado para que os dados possam ser acessados pelo container
  - Os testes são executados na montagem da imagem

# Requisitos
- Python 3.11
- pip
- Docker (opcional)
