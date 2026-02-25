from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="News Category Classifier API")

model = joblib.load("models/news_classifier.pkl")


class NewsRequest(BaseModel):
    titulo: str = Field(
        ...,
        description="Titulo da noticia para classificar.",
        examples=["Governo anuncia nova politica economica"],
    )


@app.get("/")
def read_root():
    return {"message": "News Classifier API Running"}


@app.post(
    "/readme",
    tags=["Ajuda"],
    summary="README rapido para uso da API",
    description="Retorna um guia rapido com exemplos para consumir os endpoints.",
)
def api_readme():
    return {
        "projeto": "Classificador de Categorias de Noticias",
        "como_usar": [
            "1. Verifique se a API esta rodando em http://127.0.0.1:8000",
            "2. Acesse a documentacao interativa em /docs",
            "3. Envie um POST para /predict com JSON contendo o campo titulo",
        ],
        "exemplo_requisicao": {
            "metodo": "POST",
            "endpoint": "/predict",
            "headers": {"Content-Type": "application/json"},
            "body": {"titulo": "Governo anuncia nova politica economica"},
        },
        "exemplo_resposta": {
            "categoria_predita": "mercado"
        },
        "observacoes": [
            "O campo de entrada deve se chamar titulo.",
            "A categoria retornada depende do modelo treinado em models/news_classifier.pkl.",
        ],
    }


@app.post(
    "/predict",
    tags=["Predicao"],
    summary="Classifica o titulo de uma noticia",
    description="Recebe um titulo e retorna a categoria predita pelo modelo.",
)
def predict(news: NewsRequest):
    prediction = model.predict([news.titulo])
    return {"categoria_predita": prediction[0]}
