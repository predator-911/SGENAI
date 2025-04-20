# 🖼️🔐 Text-to-Image Steganography using GenAI

> 🎨 Hide secrets inside AI-generated images like a spy 🕵️‍♂️ using the power of GenAI + Steganography!

---

## 📌 Overview

This project fuses **Generative AI** with **Steganography** to create a novel method for hiding text messages within AI-generated images. Ideal for secure communication, watermarking, and covert information transfer.

---

## 🌟 Features

- 🧠 **AI-Generated Images** using models like Stable Diffusion
- 🧵 **Text Embedding & Extraction** via custom encoding/decoding pipelines
- 🎛️ **Minimal Visual Distortion** ensures the hidden message doesn't ruin image quality
- 🛡️ **Noise Resilience** supports varying levels of image manipulation
- 🧪 **Payload Testing** with customizable text length

---

## 🧰 Tech Stack

| Tool | Usage |
|------|-------|
| 🐍 Python | Core development |
| 🧠 PyTorch / TensorFlow | AI model handling |
| 🎨 Stable Diffusion / DALL·E | Image generation |
| 🖼️ OpenCV | Image processing |
| 🌐 Streamlit | UI for demo/testing |

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/genai-steganography.git
cd genai-steganography
pip install -r requirements.txt
streamlit run app.py
