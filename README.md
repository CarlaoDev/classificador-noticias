# Classificador de Categorias de Noticias

Projeto de classificacao de noticias em categorias usando NLP (TF-IDF + LinearSVC), com exposicao de predicao por API FastAPI e execucao opcional via Docker.

## Visao geral

Este projeto contem:
- Script de treinamento (`train.py`) que le `data/articles.csv`, treina o modelo e salva em `models/news_classifier.pkl`.
- API FastAPI (`app/main.py`) para inferencia via endpoint HTTP.
- Containerizacao com Docker (`Dockerfile`).

## Estrutura do projeto

```text
.
|-- app/
|   `-- main.py
|-- data/
|   `-- articles.csv
|-- models/
|   `-- news_classifier.pkl
|-- train.py
|-- requirements.txt
|-- Dockerfile
|-- .gitignore
`-- README.md
```

## Requisitos

- Python 3.11+
- pip
- (Opcional) Docker Desktop

## Instalacao e ambiente local

### 1. Criar ambiente virtual

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Dataset

Arquivo esperado: `data/articles.csv`

Colunas obrigatorias para treino:
- `title`
- `category`

Coluna opcional (recomendada para melhor desempenho):
- `text`

Outras colunas (ex.: `date`, `subcategory`, `link`) sao ignoradas no treinamento.

## Treinamento do modelo

Execute:

```bash
python train.py
```

O script realiza:
- Validacao de colunas obrigatorias.
- Remocao de linhas com `NaN` em colunas usadas.
- Remocao de classes com menos de 2 amostras (necessario para `stratify`).
- Montagem do texto de treino com `title + text` (quando `text` existir).
- Pipeline:
  - `TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True, max_features=300000)`
  - `LinearSVC(C=1.2)`
- Impressao de `Accuracy` e `classification_report`.
- Persistencia do modelo em `models/news_classifier.pkl`.

### Resultado esperado de acuracia

Com o dataset atual e split fixo (`random_state=42`, `test_size=0.2`), a acuracia observada ficou em torno de **0.88** (pode variar em outros cenarios/dados).

## Executar a API localmente

Com o modelo ja treinado:

```bash
uvicorn app.main:app --reload
```

Acesse:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Endpoint raiz: `http://127.0.0.1:8000/`

### Endpoints

#### `GET /`
Retorna status basico da API.

#### `POST /readme`
Retorna um guia rapido em JSON com instrucoes e exemplo de uso da API.

#### `POST /predict`
Entrada:

```json
{
  "titulo": "Governo anuncia nova politica economica"
}
```

Saida (exemplo):

```json
{
  "categoria_predita": "mercado"
}
```

Observacao: apesar do dataset usar `title`, a API recebe `titulo` no JSON de entrada.

## Executar com Docker

### 1. Build da imagem

```bash
docker build -t news-classifier-api .
```

### 2. Subir container

```bash
docker run --name news-classifier-api -p 8000:8000 news-classifier-api
```

### 3. Testar

- `http://localhost:8000/docs`
- `POST /readme`
- `POST /predict` com payload JSON

### 4. Parar/remover

No terminal em execucao: `Ctrl + C`

Depois:

```bash
docker rm news-classifier-api
```

## Publicacao no GitHub

Este repositorio esta configurado para **nao versionar**:
- `data/` (dataset grande)
- `models/` (artefato do modelo treinado)
- `venv/` e caches locais

Antes de rodar a API/treino em outra maquina, garanta:
1. Colocar o dataset em `data/articles.csv`.
2. Treinar com `python train.py` para gerar `models/news_classifier.pkl`.

Se quiser versionar dataset/modelo, use Git LFS.

## Testes rapidos via PowerShell

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/ -Method Get
```

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/readme -Method Post
```

```powershell
$body = @{ titulo = "Governo anuncia nova politica economica" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

## Problemas comuns

- `KeyError: ['titulo', 'categoria']` no treino:
  - O CSV precisa usar `title` e `category` (nao `titulo`/`categoria`).

- Erro de `stratify` com classes de 1 item:
  - O `train.py` remove classes com menos de 2 amostras.

- Conflito de nome de container Docker:
  - Remova o antigo com `docker rm -f news-classifier-api` e rode novamente.

## Dependencias principais

- `pandas`, `numpy`
- `scikit-learn`
- `fastapi`, `uvicorn`
- `joblib`
