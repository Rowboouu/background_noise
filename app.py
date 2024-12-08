import streamlit as st
import noisereduce as nr
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import io  # For in-memory audio storage

# Function to load audio
def load_audio(file):
    audio, sr = librosa.load(file, sr=None)  # sr=None preserves the original sample rate
    return audio, sr

# Function to apply noise reduction
def noise_reduction(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

# Function to save the processed audio
def save_audio(audio, sr, output_path):
    sf.write(output_path, audio, sr)

# Function to plot waveform
def plot_waveform(original_audio, reduced_audio, sr):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].set_title("Original Audio Waveform")
    librosa.display.waveshow(original_audio, sr=sr, ax=axs[0])
    axs[1].set_title("Filtered Audio Waveform")
    librosa.display.waveshow(reduced_audio, sr=sr, ax=axs[1])
    plt.tight_layout()
    return fig

# Function to plot spectrogram
def plot_spectrogram(original_audio, reduced_audio, sr):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].set_title("Original Audio Spectrogram")
    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    img1 = librosa.display.specshow(D_original, sr=sr, x_axis='time', y_axis='log', ax=axs[0])
    fig.colorbar(img1, ax=axs[0], format='%+2.0f dB')

    axs[1].set_title("Filtered Audio Spectrogram")
    D_filtered = librosa.amplitude_to_db(np.abs(librosa.stft(reduced_audio)), ref=np.max)
    img2 = librosa.display.specshow(D_filtered, sr=sr, x_axis='time', y_axis='log', ax=axs[1])
    fig.colorbar(img2, ax=axs[1], format='%+2.0f dB')

    plt.tight_layout()
    return fig

# Streamlit UI setup
st.title("Background Noise Removal")
st.write("Upload an audio file to remove background noise and focus on the speaker.")

# Upload audio file
uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    # Load the audio file
    audio, sr = load_audio(uploaded_file)
    
    # Display original audio
    st.audio(uploaded_file.read(), format="audio/wav")  # Convert to bytes for st.audio
    
    # Apply noise reduction
    st.write("Processing the audio...")
    reduced_audio = noise_reduction(audio, sr)
    
    # Display the filtered audio
    st.write("Here's the audio with reduced background noise:")
    output_buffer = io.BytesIO()
    sf.write(output_buffer, reduced_audio, sr, format="WAV")
    output_buffer.seek(0)
    st.audio(output_buffer.read(), format="audio/wav")  # Convert processed audio to bytes
    
    # Plot waveform comparison
    st.write("Waveform Comparison:")
    waveform_fig = plot_waveform(audio, reduced_audio, sr)
    st.pyplot(waveform_fig)
    
    # Plot spectrogram comparison
    st.write("Spectrogram Comparison:")
    spectrogram_fig = plot_spectrogram(audio, reduced_audio, sr)
    st.pyplot(spectrogram_fig)
    
    # Save and provide download link
    output_path = "output_audio.wav"
    save_audio(reduced_audio, sr, output_path)
    with open(output_path, "rb") as file:
        st.download_button("Download Cleaned Audio", file, file_name="output_audio.wav")
