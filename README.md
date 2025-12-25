# Intraoperative Hypotension Early Warning System (IHEWS)

This repository contains a lightweight, clinically oriented prototype for **early prediction of intraoperative hypotension** during **laparoscopic cholecystectomy under general anesthesia**.

The system is designed for **practical testing by anesthesiologists** and supports both **manual data entry** and **CSV-based input**.

---

## 🔬 Clinical Task
- **Target:** Intraoperative hypotension
- **Definition:** MAP < 65 mmHg sustained for ≥ 60 seconds
- **Prediction horizon:** 10 minutes
- **Procedure:** Laparoscopic cholecystectomy (GA)

---

## ⚙️ Model
- Interpretable **Logistic Regression**
- Inputs:
  - MAP
  - HR
  - SpO₂
  - RR
  - EtCO₂
- Trained on real intraoperative data from VitalDB
- Sensitivity-oriented operating point (92%)

---

## 🚨 Alarm Management
- Refractory alarm strategy (user-configurable)
- Typical alert rate:
  - ~2 alerts/hour (5-minute refractory)
  - ~1.5 alerts/hour (10-minute refractory)

---

## 🖥️ Application Features
- Manual data entry (realistic OR workflow)
- CSV upload for retrospective testing
- User-defined time axis (manual / auto / real-time)
- Real-time risk probability display
- Alarm visualization and logging
- Exportable session logs

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
