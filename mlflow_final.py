from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from xgboost import XGBClassifier


BASE_DIR = Path("datasets/dadosprocessados")

X_TREINO_PATH = BASE_DIR / "X_treino.csv"
X_TESTE_PATH = BASE_DIR / "X_teste.csv"
Y_TREINO_PATH = BASE_DIR / "y_treino.csv"
Y_TESTE_PATH = BASE_DIR / "y_teste.csv"

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def carregar_x(caminho):
    df = pd.read_csv(caminho)

    # Remove coluna de índice caso exista
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    return df


def carregar_y(caminho):
    df = pd.read_csv(caminho)

    # Remove coluna de índice caso exista
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Transforma DataFrame de uma coluna em Series
    y = df.squeeze()

    return y.astype(int)


def carregar_dados_processados():
    X_treino = carregar_x(X_TREINO_PATH)
    X_teste = carregar_x(X_TESTE_PATH)

    y_treino = carregar_y(Y_TREINO_PATH)
    y_teste = carregar_y(Y_TESTE_PATH)

    return X_treino, X_teste, y_treino, y_teste


def obter_score(modelo, X_teste):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X_teste)[:, 1]

    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X_teste)

    return None


def registrar_experimento(nome_modelo, modelo, parametros, X_treino, X_teste, y_treino, y_teste):
    with mlflow.start_run(run_name=nome_modelo):
        mlflow.log_param("nome_modelo", nome_modelo)
        mlflow.log_param("target", "Potability")
        mlflow.log_param("classe_positiva", "1 = potável")
        mlflow.log_param("classe_negativa", "0 = não potável")

        for chave, valor in parametros.items():
            mlflow.log_param(chave, valor)

        modelo.fit(X_treino, y_treino)

        y_pred = modelo.predict(X_teste)
        y_score = obter_score(modelo, X_teste)

        matriz = confusion_matrix(y_teste, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matriz.ravel()

        metricas = {
            "accuracy": accuracy_score(y_teste, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_teste, y_pred),
            "precision_potable": precision_score(y_teste, y_pred, pos_label=1, zero_division=0),
            "recall_potable": recall_score(y_teste, y_pred, pos_label=1, zero_division=0),
            "f1_potable": f1_score(y_teste, y_pred, pos_label=1, zero_division=0),
            "f1_macro": f1_score(y_teste, y_pred, average="macro", zero_division=0),
            "falso_potavel": fp,
            "falso_nao_potavel": fn,
        }

        if y_score is not None:
            metricas["roc_auc"] = roc_auc_score(y_teste, y_score)

        for nome_metrica, valor in metricas.items():
            mlflow.log_metric(nome_metrica, float(valor))

        relatorio = classification_report(
            y_teste,
            y_pred,
            target_names=["Não potável", "Potável"],
            output_dict=True,
            zero_division=0
        )

        caminho_relatorio = REPORTS_DIR / f"relatorio_{nome_modelo}.json"

        with open(caminho_relatorio, "w", encoding="utf-8") as arquivo:
            json.dump(relatorio, arquivo, indent=4, ensure_ascii=False)

        mlflow.log_artifact(str(caminho_relatorio))

        display = ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=["Não potável", "Potável"]
        )

        display.plot(values_format="d")
        plt.title(f"Matriz de Confusão - {nome_modelo}")
        plt.tight_layout()

        caminho_matriz = REPORTS_DIR / f"matriz_confusao_{nome_modelo}.png"
        plt.savefig(caminho_matriz, dpi=150)
        plt.close()

        mlflow.log_artifact(str(caminho_matriz))

        try:
            mlflow.sklearn.log_model(
                sk_model=modelo,
                name="modelo",
                serialization_format="cloudpickle"
            )
        except Exception as erro:
            print(f"Aviso: não foi possível salvar o modelo {nome_modelo} no MLflow.")
            print(f"Erro ao salvar modelo: {erro}")
            mlflow.log_text(
                str(erro),
                f"erro_salvar_modelo_{nome_modelo}.txt"
            )

        return {
            "modelo": nome_modelo,
            **metricas
        }


def main():
    X_treino, X_teste, y_treino, y_teste = carregar_dados_processados()

    print("Dados carregados com sucesso.")
    print(f"X_treino: {X_treino.shape}")
    print(f"X_teste: {X_teste.shape}")
    print(f"y_treino: {y_treino.shape}")
    print(f"y_teste: {y_teste.shape}")

    mlflow.set_experiment("water-potability-comparacao-final")

    modelos = [
        (
            "dummy_baseline",
            DummyClassifier(strategy="most_frequent"),
            {
                "responsavel": "Bryan",
                "modelo": "DummyClassifier",
                "strategy": "most_frequent"
            }
        ),

        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            {
                "responsavel": "Equipe",
                "modelo": "LogisticRegression",
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42
            }
        ),

        (
            "decision_tree",
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {
                "responsavel": "Equipe",
                "modelo": "DecisionTreeClassifier",
                "class_weight": "balanced",
                "random_state": 42
            }
        ),

        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            ),
            {
                "responsavel": "Equipe",
                "modelo": "RandomForestClassifier",
                "n_estimators": 100,
                "class_weight": "balanced",
                "random_state": 42
            }
        ),

        (
            "svm_rbf",
            SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=42
            ),
            {
                "responsavel": "Equipe",
                "modelo": "SVC",
                "kernel": "rbf",
                "C": 1.0,
                "probability": True,
                "class_weight": "balanced",
                "random_state": 42
            }
        ),

        (
            "knn",
            KNeighborsClassifier(n_neighbors=5),
            {
                "responsavel": "Equipe",
                "modelo": "KNeighborsClassifier",
                "n_neighbors": 5
            }
        ),

        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=42),
            {
                "responsavel": "Equipe",
                "modelo": "GradientBoostingClassifier",
                "random_state": 42
            }
        ),

        (
            "xgboost",
            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                eval_metric="logloss",
                random_state=42
            ),
            {
                "responsavel": "Equipe",
                "modelo": "XGBClassifier",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "eval_metric": "logloss",
                "random_state": 42
            }
        ),
    ]

    resultados = []

    for nome_modelo, modelo, parametros in modelos:
        print(f"\nRodando experimento: {nome_modelo}")

        resultado = registrar_experimento(
            nome_modelo=nome_modelo,
            modelo=modelo,
            parametros=parametros,
            X_treino=X_treino,
            X_teste=X_teste,
            y_treino=y_treino,
            y_teste=y_teste
        )

        resultados.append(resultado)

    df_resultados = pd.DataFrame(resultados)

    colunas = [
        "modelo",
        "accuracy",
        "balanced_accuracy",
        "precision_potable",
        "recall_potable",
        "f1_potable",
        "f1_macro",
        "roc_auc",
        "falso_potavel",
        "falso_nao_potavel"
    ]

    colunas_existentes = [coluna for coluna in colunas if coluna in df_resultados.columns]
    df_resultados = df_resultados[colunas_existentes]

    df_resultados = df_resultados.sort_values(
        by=["f1_macro", "roc_auc"],
        ascending=False
    )

    caminho_comparacao = REPORTS_DIR / "comparacao_final_modelos.csv"
    df_resultados.to_csv(caminho_comparacao, index=False)

    print("\nComparação final dos modelos:")
    print(df_resultados)

    print(f"\nArquivo salvo em: {caminho_comparacao}")


if __name__ == "__main__":
    main()