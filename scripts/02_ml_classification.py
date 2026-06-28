#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Machine Learning para Predição de Resposta à Imunoterapia
Descrição: Carrega dados processados pelo R, treina modelos de classificação 
           (Random Forest e XGBoost), avalia a performance e exporta todas as
           figuras analíticas separadamente em alta resolução.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

def main():
    print("[-] Iniciando pipeline de Machine Learning...")
    
    input_path = "/home/nara/projects/GSE236581/Single-cell/scripts/processed_expression_for_ml.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo {input_path} não encontrado. Por favor, execute o script em R primeiro.")
        
    # 1. Carregamento dos dados processados
    df = pd.read_csv(input_path)
    print(f"[+] Dataset carregado: {df.shape[0]} amostras e {df.shape[1]} colunas.")
    
    # 2. Separação de Features (X) e Target (y)
    y = df['Response'].map({'Responder': 1, 'Non-responder': 0}).values
    X = df.drop(columns=['Patient_ID', 'Response', 'Cohort'])
    
    gene_names = X.columns.tolist()
    
    # 3. Divisão em Treino e Teste (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"[+] Divisão Treino/Teste concluída (Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]}).")
    
    # ==========================================
    # MODELO 1: Random Forest Classifier
    # ==========================================
    print("\n[-] Treinando Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    print("=== Resultados: Random Forest ===")
    print(f"Acurácia: {accuracy_score(y_test, rf_preds):.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.2f}")
    print(classification_report(y_test, rf_preds, target_names=['Non-responder', 'Responder']))
    
    # ==========================================
    # MODELO 2: XGBoost Classifier
    # ==========================================
    print("\n[-] Treinando XGBoost...")
    xgb_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    print("=== Resultados: XGBoost ===")
    print(f"Acurácia: {accuracy_score(y_test, xgb_preds):.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.2f}")
    print(classification_report(y_test, xgb_preds, target_names=['Non-responder', 'Responder']))
    
    # Extração de Importância dos Genes
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n=== Top 5 Genes Mais Importantes (Random Forest) ===")
    for f in range(5):
        print(f"{f + 1}. Gene: {gene_names[indices[f]]} ({importances[indices[f]]*100:.2f}%)")
        
    print("\n[SUCCESS] Modelagem concluída. Iniciando exportação de figuras separadas...")

    # ==========================================
    # GERAÇÃO E EXPORTAÇÃO DE FIGURAS INDIVIDUAIS
    # ==========================================
    figs_path = "figures"
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)
        
    # Calcular métricas para Curva ROC
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)
    
    # ------------------------------------------
    # FIGURA 1: Curvas ROC (Model Performance)
    # ------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)', alpha=0.7)
    plt.plot(xgb_fpr, xgb_tpr, color='#1f77b4', lw=2, label=f'XGBoost (AUC = {xgb_auc:.2f})')
    plt.plot(rf_fpr, rf_tpr, color='#2ca02c', lw=2, label=f'Random Forest (AUC = {rf_auc:.2f})')
    plt.title("Model Performance Comparison (ROC Curves)", fontsize=14, pad=15)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_path, "roc_curves.png"), dpi=300)
    plt.close()
    print("[+] Figura 'roc_curves.png' salva com sucesso.")

    # ------------------------------------------
    # FIGURA 2: Clustered Heatmap (Seaborn)
    # ------------------------------------------
    heatmap_df = df.set_index('Patient_ID').drop(columns=['Cohort'])
    # Mapeamento para barra de cores de anotação clínica
    col_colors = heatmap_df.pop('Response').map({'Responder': 'teal', 'Non-responder': 'orange'})
    
    g = sns.clustermap(
        heatmap_df, 
        cmap='coolwarm', 
        figsize=(8, 8), 
        col_colors=col_colors, 
        yticklabels=False,
        cbar_kws={'label': 'Expression Level (Log2)'}
    )
    g.fig.suptitle("Gene Expression Clustered Heatmap", y=1.02, fontsize=14)
    plt.savefig(os.path.join(figs_path, "heatmap_clustered.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Figura 'heatmap_clustered.png' salva com sucesso.")
    
    # ------------------------------------------
    # FIGURA 3: Feature Importance Barplot
    # ------------------------------------------
    plt.figure(figsize=(8, 5))
    feat_importances = pd.Series(rf_model.feature_importances_, index=gene_names)
    feat_importances.nlargest(5).sort_values(ascending=True).plot(
        kind='barh', 
        color='#2a9d8f', 
        edgecolor='black',
        alpha=0.85
    )
    plt.title("Top 5 Feature Importance (Random Forest Model)", fontsize=14, pad=15)
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Biomarker Candidate (Gene)", fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_path, "feature_importance.png"), dpi=300)
    plt.close()
    print("[+] Figura 'feature_importance.png' salva com sucesso.")
    
    # ------------------------------------------
    # FIGURA 4: Cohort-Specific Boxplot (CD8A)
    # ------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x='Response', 
        y='CD8A', 
        hue='Cohort', 
        data=df, 
        palette='Set2',
        linewidth=1.5
    )
    plt.title("CD8A Expression Across Cohorts and Clinical Response", fontsize=14, pad=15)
    plt.xlabel("Immunotherapy Response Status", fontsize=12)
    plt.ylabel("Log2 Normalized Expression", fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.legend(title="Integrated Cohorts", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_path, "cohort_expression_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Figura 'cohort_expression_boxplot.png' salva com sucesso.")
    
    print(f"\n[SUCCESS] Todas as figuras foram exportadas individualmente para '{figs_path}/'!")

if __name__ == "__main__":
    main()
