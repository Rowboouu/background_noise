import streamlit as st
import noisereduce as nr
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import tempfile  # For temporary file storage

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
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].set_title("Original Audio Waveform")
    librosa.display.waveshow(original_audio, sr=sr, ax=ax[0])
    ax[1].set_title("Filtered Audio Waveform")
    librosa.display.waveshow(reduced_audio, sr=sr, ax=ax[1])
    st.pyplot(fig)  # Pass the figure to st.pyplot()

# Function to plot spectrogram
def plot_spectrogram(original_audio, reduced_audio, sr):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Original audio spectrogram
    ax[0].set_title("Original Audio Spectrogram")
    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    img_original = librosa.display.specshow(D_original, sr=sr, x_axis='time', y_axis='log', ax=ax[0])
    fig.colorbar(img_original, ax=ax[0], format='%+2.0f dB')  # Associate colorbar with spectrogram
    
    # Filtered audio spectrogram
    ax[1].set_title("Filtered Audio Spectrogram")
    D_filtered = librosa.amplitude_to_db(np.abs(librosa.stft(reduced_audio)), ref=np.max)
    img_filtered = librosa.display.specshow(D_filtered, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    fig.colorbar(img_filtered, ax=ax[1], format='%+2.0f dB')  # Associate colorbar with spectrogram
    
    st.pyplot(fig)  # Pass the figure to st.pyplot()


# Streamlit UI setup
st.title("Background Noise Removal")
st.markdown("**Concillo | Go | Moncano | Paring**")
st.markdown("**CPE416 - Final PIT**")

# Upload audio file
uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display the original audio directly from the uploaded file
    st.write("Original Audio:")
    st.audio(uploaded_file)  # Use the raw uploaded file directly for playback

    # Load the audio for processing
    audio, sr = load_audio(uploaded_file)
    
    # Apply noise reduction
    reduced_audio = noise_reduction(audio, sr)
    
    # Save the processed audio to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    save_audio(reduced_audio, sr, temp_file.name)
    
    # Display the filtered audio
    st.write("Filtered Audio with Reduced Background Noise:")
    st.audio(temp_file.name)  # Use the saved file path directly for playback
    
    # Plot waveform comparison
    plot_waveform(audio, reduced_audio, sr)
    
    # Plot spectrogram comparison
    plot_spectrogram(audio, reduced_audio, sr)
    
    # Save and provide download link
    output_path = "output_audio.wav"
    save_audio(reduced_audio, sr, output_path)
    st.download_button("Download Cleaned Audio", data=open(output_path, "rb"), file_name="output_audio.wav")
