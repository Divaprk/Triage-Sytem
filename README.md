# AI-Powered Edge Triage System (Team G35)

## 📌 Project Overview
The AI-Powered Edge Triage System is an automated medical assessment tool designed for the **INF2009: Edge Computing & Analytics** module[cite: 1, 2]. The system aims to standardize and accelerate the patient prioritization process in Emergency Rooms or hospital wards by performing real-time analytics at the edge.

Using a **Raspberry Pi 5**, the system fuses biometric data, verbal complaints, and visual indicators to classify patients into Patient Acuity Categories (PAC) 1 through 4[cite: 25, 41, 58, 59, 60].

---

## 🏗️ Core Features
The project integrates three primary data streams for a comprehensive patient assessment:

* **Vitals:** Heart rate, $SPO_2$, and body temperature collected via wearable devices (e.g., Samsung Galaxy Watch or Fitbit) or backup sensors.
* **Speech:** Local Speech-to-Text (STT) and LLM inference to extract the patient's "Chief Complaint".
* **Vision:** Computer vision to detect physical characteristics like eye dilation, sweating, or abnormal movement. (Undecided)

---

## 📂 Project Structure
Following the required work packages, the project is organized as follows:

* `/docs`: Project documentation, hardware justifications, and meeting notes.
* `/vitals`: Biometric data acquisition and sensor integration.
* `/speech`: Audio processing and local LLM logic.
* `/vision`: Camera-based symptom detection.
* `/dashboard`: User interface for medical staff to view real-time triage results[cite: 233, 234].

---

## 👥 The Team
* **Alexi George**
* **Cavell Lim**
* **Lin Yu Chuan**
* **Bryan Law**
* **Divakaran Prakash**