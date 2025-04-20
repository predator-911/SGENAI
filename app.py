# Install required libraries (Uncomment and run if needed)
# !pip install streamlit diffusers torch torchvision transformers numpy pillow cryptography opencv-python

import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import io
import cv2
from cryptography.fernet import Fernet

# Load AES Key from File (Persistent)
@st.cache_resource
def load_aes_key():
    try:
        with open("aes_key.txt", "rb") as key_file:
            return key_file.read()
    except FileNotFoundError:
        key = Fernet.generate_key()
        with open("aes_key.txt", "wb") as key_file:
            key_file.write(key)
        return key

aes_key = load_aes_key()  # Load the AES key globally

# Load Stable Diffusion Model Efficiently
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Generate Image from Text
def generate_image_from_text(pipe, prompt):
    with st.spinner("Generating image..."):
        result = pipe(prompt, guidance_scale=7.5)
        image = result.images[0]
    return image

# Encrypt Message using AES
def encrypt_message(secret_text, key):
    cipher = Fernet(key)
    return cipher.encrypt(secret_text.encode()).decode()

# Decrypt Message using AES
def decrypt_message(encrypted_text, key):
    cipher = Fernet(key)
    return cipher.decrypt(encrypted_text.encode()).decode()

# Embed Text in Image (with Color Channel Option)
def encode_text_in_image(image, secret_text, channel='B'):
    binary_text = ''.join(format(ord(c), '08b') for c in secret_text) + '11111111'  # End marker
    pixels = np.array(image, dtype=np.uint8)

    # Select color channel
    channel_index = {'R': 0, 'G': 1, 'B': 2}[channel]
    selected_channel = pixels[:, :, channel_index].copy()

    # Check max capacity
    max_chars = (selected_channel.size // 8) - 1
    if len(secret_text) > max_chars:
        st.error(f"The secret message is too long! Maximum allowed: {max_chars} characters")
        return None

    # Embed in LSB
    for i, bit in enumerate(binary_text):
        selected_channel.flat[i] = (selected_channel.flat[i] & 0b11111110) | int(bit)

    # Update the image
    pixels[:, :, channel_index] = selected_channel
    return Image.fromarray(pixels)

# Decode Text from Image
def decode_text_from_image(image, channel='B'):
    pixels = np.array(image, dtype=np.uint8)
    channel_index = {'R': 0, 'G': 1, 'B': 2}[channel]
    selected_channel = pixels[:, :, channel_index]

    binary_text = ''.join(str(pixel & 1) for pixel in selected_channel.flatten())

    hidden_message = []
    for i in range(0, len(binary_text), 8):
        byte = binary_text[i:i + 8]
        if byte == '11111111':
            break
        hidden_message.append(chr(int(byte, 2)))

    return ''.join(hidden_message)

# Embed a Watermark
def add_watermark(image, text="Steganography App"):
    watermark = image.copy()
    draw = ImageDraw.Draw(watermark)
    font = ImageFont.load_default()
    width, height = image.size
    draw.text((width - 150, height - 30), text, font=font, fill=(255, 255, 255, 128))
    return watermark

# Embed an Image Inside Another Image (Image-in-Image)
def embed_image_into_image(base_image, hidden_image):
    base = np.array(base_image.convert("RGBA"))
    hidden = np.array(hidden_image.resize(base_image.size).convert("RGBA"))

    # Blend the images
    alpha = 0.3  # Adjust transparency
    blended = cv2.addWeighted(base, 1, hidden, alpha, 0)
    
    return Image.fromarray(blended)

# Streamlit App
def main():
    st.title("Text-to-Image Steganography Web App")
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Choose an option", ["Home", "Generate & Embed", "Generate Image", "Embed Secret Text", "Decode Secret Text", "Embed Image"])

    if option == "Home":
        st.subheader("Welcome!")
        st.write("""
        - **Generate an image** using AI (Stable Diffusion).
        - **Embed secret text** inside an image using AES encryption and steganography.
        - **Embed another image inside an image.**
        - **Decode hidden messages** from images.
        """)

    elif option == "Generate & Embed":
        st.subheader("Generate Image and Embed Secret Text")
        pipe = load_pipeline()

        prompt = st.text_input("Enter a text prompt", "A futuristic cityscape at sunset.")
        secret_message = st.text_input("Enter the secret message to embed")
        channel = st.selectbox("Choose Color Channel", ["R", "G", "B"])

        if st.button("Generate & Embed"):
            image = generate_image_from_text(pipe, prompt)

            if secret_message:
                encrypted_text = encrypt_message(secret_message, aes_key)
                stego_image = encode_text_in_image(image, encrypted_text, channel)
                stego_image = add_watermark(stego_image)

                st.image(stego_image, caption="Stego Image with Secret Message", use_column_width=True)
                img_byte_arr = io.BytesIO()
                stego_image.save(img_byte_arr, format="PNG")
                st.download_button("Download Stego Image", img_byte_arr.getvalue(), file_name="stego_image.png", mime="image/png")

    elif option == "Embed Secret Text":
        st.subheader("Embed a Secret Message")
        uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        secret_message = st.text_input("Enter the secret message")
        channel = st.selectbox("Choose Color Channel", ["R", "G", "B"])

        if st.button("Embed Message"):
            if uploaded_image and secret_message:
                image = Image.open(uploaded_image).convert("RGB")
                encrypted_text = encrypt_message(secret_message, aes_key)
                stego_image = encode_text_in_image(image, encrypted_text, channel)
                stego_image = add_watermark(stego_image)

                st.image(stego_image, caption="Stego Image", use_column_width=True)
                img_byte_arr = io.BytesIO()
                stego_image.save(img_byte_arr, format="PNG")
                st.download_button("Download Stego Image", img_byte_arr.getvalue(), file_name="stego_image.png", mime="image/png")

    elif option == "Decode Secret Text":
        st.subheader("Decode a Secret Message")
        uploaded_image = st.file_uploader("Upload a Stego Image", type=["png", "jpg", "jpeg"])
        channel = st.selectbox("Choose Color Channel", ["R", "G", "B"])

        if st.button("Decode Message"):
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                decoded_message = decode_text_from_image(image, channel)
                
                try:
                    decrypted_message = decrypt_message(decoded_message, aes_key)
                    st.success(f"Decoded Message: {decrypted_message}")
                except Exception:
                    st.error("Decryption failed! The key may be incorrect or the image does not contain valid encrypted data.")

    elif option == "Embed Image":
        st.subheader("Embed an Image into Another Image")
        base_image = st.file_uploader("Upload Base Image", type=["png", "jpg", "jpeg"])
        hidden_image = st.file_uploader("Upload Hidden Image", type=["png", "jpg", "jpeg"])

        if st.button("Embed Image"):
            if base_image and hidden_image:
                base = Image.open(base_image)
                hidden = Image.open(hidden_image)
                blended = embed_image_into_image(base, hidden)
                st.image(blended, caption="Image with Hidden Image", use_column_width=True)

if __name__ == "__main__":
    main()
    

