# Use uma imagem base oficial do Python
FROM python:3.11-slim

# Define o diretório de trabalho no container
WORKDIR /app

COPY requirements.txt .

# Instala pandas e quaisquer outras dependências necessárias
RUN pip install -r requirements.txt

# Copia os arquivos do seu projeto para o diretório de trabalho no container
COPY . .

# Comando para executar o script quando o container iniciar
CMD ["python", "./main.py"]
