# Customer Churn Prediction with RAG-Based Explanations

## Overview
This project focuses on predicting customer churn using multiple ML and DL models, while providing **interpretable explanations** for predictions via a Retrieval-Augmented Generation (RAG) system. The goal is to help business stakeholders understand why a customer is likely to leave, based on patterns observed in past customers.

## Problem
Customer churn is costly for subscription-based services. While ML models can predict churn with high accuracy, **understanding the reasons behind predictions is critical** for retention strategies. Traditional explainability methods (like SHAP) provide feature importance, but **natural-language explanations grounded in real customer cases** can improve interpretability and trust.

## Approach
- Handled **unbalanced dataset** using SMOTE to oversample the minority (churn) class
- Trained and evaluated multiple ML models (Logistic Regression, Random Forest, XGBoost) for churn prediction
- Generated SHAP-based feature importances for each prediction
- Built a RAG-based explanation system:
  - Retrieved similar past customer cases + relevant passages
  - Used an LLM (Mistral 7B Instruct) to generate natural-language explanations grounded in retrieved examples
- Created structured prompts to guide the LLM to produce concise, accurate, and relevant explanations
- Focused on interpretable output rather than just predictive accuracy

## Technologies
- Python  
- scikit-learn, PyTorch  
- SHAP (for feature importance)  
- Hugging Face Transformers  
- RAG pipeline (retrieval + LLM generation)  

## Results
- ML models achieved balanced predictive performance (e.g., XGBoost ≈ **0.84 AUC**)  
- RAG system produced **coherent explanations** reflecting model reasoning and observed patterns  
- Demonstrated feasibility of **using retrieval to ground explanations in past cases**  
- Highlighted important churn factors such as **tenure, contract type, payment method, and online security**  

## Evaluation
- **Intrinsic evaluation:** Fluency, coherence, and alignment with SHAP-based reasoning  
- **Consistency checks:** Similar queries yielded consistent explanations  
- **Human-in-the-loop:** Verified readability and usefulness of explanations 

## Repository Notes  
- All experiments can be reproduced using the provided training, evaluation, and explanation scripts  

## Key Skills Demonstrated
- Machine learning and deep learning for churn prediction  
- Explainable AI with RAG and SHAP  
- Designing interpretable, business-relevant outputs from ML predictions  
- Human-in-the-loop evaluation for unsupervised explanation assessment
