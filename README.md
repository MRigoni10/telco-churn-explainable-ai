# 📉 Early Warning Churn System con XGBoost & Explainable AI (SHAP)

## 📌 Executive Summary
Acquisire nuovi clienti costa da 5 a 7 volte più che trattenerne uno esistente. Questo progetto implementa un sistema predittivo per identificare i clienti a rischio abbandono con almeno 30 giorni di anticipo e ne diagnostica le cause scatenanti tramite Explainable AI (SHAP), permettendo azioni mirate di Customer Success.

## 🛠️ Architettura Tecnica
- **Dataset:** IBM Telco Customer Churn (7.000+ clienti).
- **Modello:** XGBoost Classifier con bilanciamento classi tramite `scale_pos_weight`.
- **Explainability:** SHAP TreeExplainer per feature importance globale e waterfall plots per diagnosi del singolo cliente.
- **Metriche:** ROC-AUC: 0.84 | PR-AUC: 0.65.

## 💡 Business Impact & Risultati
- **Fattori Critici di Churn:** Contratti mensili (*Month-to-Month*), metodo di pagamento con *Electronic Check* e assenza di servizi di supporto tecnico (*TechSupport*).
- **Simulazione ROI:** Su un campione di test di ~1.400 clienti, il modello consente di recuperare fino al 40% dei clienti a rischio, generando un profitto netto stimato di oltre €15.000 al netto dei costi di retention.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TiglJvX3sAcY-IfouEEYokVW26zvqhwV)
