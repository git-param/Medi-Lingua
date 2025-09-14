# 🩺 MediLingua - Multilingual Medical Support Assistant

## 🌍 Project Overview
MediLingua is a **text-based multilingual medical support assistant** designed to help rural and low-resource patients describe symptoms in their **native language**.  
The system translates the text into English, extracts key medical entities, summarizes the input, classifies possible conditions, clusters similar cases, and produces a **structured report** for healthcare workers.  
This addresses the communication gap in rural healthcare where patients may not speak English.

---

## 🚀 Features
- 🌐 **Translation**: Converts Indian regional languages (e.g., Gujarati, Hindi) into English using **IndicTrans2**.  
- 🧾 **Entity Extraction**: Identifies symptoms, medicines, and conditions with **NER models**.  
- ✍️ **Summarization**: Creates concise symptom summaries using **T5 / BART**.  
- 🩻 **Classification & Clustering**: Maps symptoms to likely conditions and groups similar patient cases.  
- 📑 **Report Generation**: Produces structured outputs in JSON/PDF for doctors and NGOs.  

---

## 🛠️ Tech Stack
- **Programming**: Python  
- **NLP**: spaCy, Stanza, HuggingFace Transformers  
- **Machine Translation**: IndicTrans2  
- **Summarization**: T5, BART  
- **Classification**: BioBERT, FastText  
- **Clustering**: scikit-learn  
- **Web/UI**: Flask, Streamlit  

---

## 📂 Datasets & Resources
-	https://www.kaggle.com/datasets/yousefsaeedian/ai-medical-chatbot?resource=download<img width="468" height="60" alt="image" src="https://github.com/user-attachments/assets/deecfc51-dad5-431e-aa71-c56161447f5d" />

---

## 📊 Evaluation
- **Translation**: BLEU, COMET  
- **NER**: Precision, Recall, F1  
- **Summarization**: ROUGE  
- **Classification**: Accuracy/F1  
- **Clustering**: Coherence  

---

## 🌟 Why MediLingua?
- Socially impactful: improves rural healthcare communication  
- Complete NLP pipeline from **translation to clustering**  
- Novel use of **multilingual resources** for medical AI  

---

## 📜 License
This project is licensed under the **MIT License**.
