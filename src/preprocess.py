import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def carregar_dados(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado no caminho: {filepath}")
        return None

def separar_treino_teste(df, target_col='Potability'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def criar_pipeline_pre_processamento():
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline

def executar_pipeline_completo(filepath):
    print("Iniciando pré-processamento...")
    df = carregar_dados(filepath)
    if df is None:
        return None, None, None, None
        
    X_train, X_test, y_train, y_test = separar_treino_teste(df)
    
    pipeline = criar_pipeline_pre_processamento()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    colunas = X_train.columns
    X_train_processed = pd.DataFrame(X_train_processed, columns=colunas)
    X_test_processed = pd.DataFrame(X_test_processed, columns=colunas)
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)
    
    print("[OK] Pré-processamento concluído com sucesso!")
    print(f" -> Treino (Balanceado com SMOTE): {X_train_bal.shape[0]} amostras")
    print(f" -> Teste (Original Padronizado): {X_test_processed.shape[0]} amostras")
    
    return X_train_bal, X_test_processed, y_train_bal, y_test
