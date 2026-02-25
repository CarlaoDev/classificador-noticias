import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def main():
    # Carrega o dataset
    df = pd.read_csv("data/articles.csv")

    # Limpeza basica e validacao do esquema
    required_cols = ["title", "category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatorias ausentes no CSV: {missing}. "
            f"Colunas encontradas: {list(df.columns)}"
        )

    # Usa title + text quando disponivel para melhorar a qualidade da classificacao
    text_cols = ["title", "category"]
    if "text" in df.columns:
        text_cols.append("text")

    df = df.dropna(subset=text_cols)

    # O split estratificado exige pelo menos 2 amostras por classe
    class_counts = df["category"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df["category"].isin(valid_classes)]

    if "text" in df.columns:
        X = df["title"].astype(str) + " " + df["text"].astype(str)
    else:
        X = df["title"].astype(str)
    y = df["category"]

    # Separacao treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline ajustado para texto esparso em grande volume e melhor acuracia
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                sublinear_tf=True,
                max_features=300000,
            ),
        ),
        ("clf", LinearSVC(C=1.2)),
    ])

    # Treinamento
    model.fit(X_train, y_train)

    # Avaliacao
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Salva o modelo
    joblib.dump(model, "models/news_classifier.pkl")
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
