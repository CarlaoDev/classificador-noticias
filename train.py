import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def detectar_colunas_data(df: pd.DataFrame) -> list[str]:
    candidatos = []
    for col in df.columns:
        nome = col.lower()
        if any(chave in nome for chave in ["date", "data"]):
            candidatos.append(col)
    return candidatos


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Shape inicial: {df.shape}")

    # 2) Diagnostico de dados faltantes
    faltantes = df.isna().mean().sort_values(ascending=False)
    faltantes = faltantes[faltantes > 0]
    if not faltantes.empty:
        print("\nColunas com valores faltantes (%):")
        print((faltantes * 100).round(2).to_string())

    # 3) Remove colunas muito vazias (exceto colunas protegidas)
    protegidas = {"title", "text", "category", "date"}
    cols_muito_vazias = [
        c for c in df.columns if df[c].isna().mean() >= 0.98 and c not in protegidas
    ]
    if cols_muito_vazias:
        print(f"\nRemovendo colunas muito vazias: {cols_muito_vazias}")
        df = df.drop(columns=cols_muito_vazias)

    # 4) Converte datas para datetime e cria features temporais simples
    cols_data = detectar_colunas_data(df)
    for col in cols_data:
        parsed = pd.to_datetime(df[col], errors="coerce")
        taxa_valida = parsed.notna().mean()
        if taxa_valida >= 0.8:
            print(f"Convertendo coluna '{col}' para datetime (taxa valida {taxa_valida:.2%})")
            df[col] = parsed
            base_nome = col.lower()
            df[f"{base_nome}_ano"] = df[col].dt.year
            df[f"{base_nome}_mes"] = df[col].dt.month
            df[f"{base_nome}_dia_semana"] = df[col].dt.dayofweek

    print(f"Shape apos limpeza: {df.shape}")
    return df


def main():
    # 1) Importa arquivo com read_csv
    df = pd.read_csv("data/articles.csv", low_memory=False)

    # Preparacao geral
    df = preparar_dados(df)

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

    # 5) Analisa distribuicao de classes
    class_counts = df["category"].value_counts()
    print(f"\nQuantidade de classes: {class_counts.size}")
    print("Top 10 classes:")
    print(class_counts.head(10).to_string())
    print("Bottom 10 classes:")
    print(class_counts.tail(10).to_string())

    # 6) Agrupa classes muito raras em 'outras' (opcional e controlado)
    min_amostras_classe = 5
    raras = class_counts[class_counts < min_amostras_classe]
    if not raras.empty:
        frac_raras = raras.sum() / len(df)
        print(
            f"\nClasses raras (<{min_amostras_classe} amostras): {len(raras)} "
            f"({frac_raras:.2%} do dataset)"
        )
        # Faz sentido agrupar quando impacto no volume total e pequeno
        if frac_raras <= 0.05:
            df["category"] = df["category"].where(
                ~df["category"].isin(raras.index), "outras"
            )
            print("Classes raras foram agrupadas em 'outras'.")
        else:
            print("Nao agrupado: impacto alto no dataset.")

    # Garante estratificacao valida
    class_counts = df["category"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    removidas = class_counts[class_counts < 2]
    if not removidas.empty:
        print(f"Removendo classes com menos de 2 amostras: {list(removidas.index)}")
    df = df[df["category"].isin(valid_classes)]

    # Monta texto final
    if "text" in df.columns:
        X = df["title"].astype(str) + " " + df["text"].astype(str)
    else:
        X = df["title"].astype(str)

    # Se houver features temporais, adiciona como tokens de texto
    for col in df.columns:
        if col.endswith("_ano") or col.endswith("_mes") or col.endswith("_dia_semana"):
            token_prefix = col.upper()
            X = X + " " + token_prefix + "_" + df[col].fillna(-1).astype(int).astype(str)

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
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Salva o modelo
    joblib.dump(model, "models/news_classifier.pkl")
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
