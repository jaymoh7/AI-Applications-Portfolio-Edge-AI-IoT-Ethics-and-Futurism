# AI Applications Portfolio – Edge AI, IoT, Ethics, and Futurism

## 🔍 Overview
This repository contains theoretical, practical, and visionary work on real-world AI applications. It includes a working Edge AI prototype, conceptual IoT architecture for agriculture, ethical analysis in personalized medicine, and a futuristic AI proposal for 2030.

---

## 📁 Contents

| File/Folder | Description |
|-------------|-------------|
| `edge_ai_image_classification.py` | Lightweight CNN for classifying rock/paper/scissors images. Converted to TFLite for Edge deployment. |
| `Smart_Agriculture_Dataflow_Diagram.png` | Visual data flow for AI-driven IoT system in smart agriculture. |
| `report.pdf` | Written responses for theoretical analysis, implementation reports, and futuristic proposal. |

---

## 🛠️ Part 1: Theoretical Analysis

Includes answers to:
- Edge AI vs Cloud AI (with real-world example).
- Quantum AI vs Classical AI in optimization.
- Human-AI collaboration in healthcare.
- Case study critique on AI-IoT in smart cities.

All responses are compiled in the `report.pdf`.

---

## 🧪 Part 2: Practical Implementation

### ✅ Task 1: Edge AI Image Classification
- Trains a small CNN using TensorFlow.
- Dataset: Rock-Paper-Scissors (TFDS).
- Model is converted to `.tflite` and ready for deployment to edge devices like Raspberry Pi.
- **Accuracy**: ~95% (validation)

Run the script:

```bash
python edge_ai_image_classification.py
