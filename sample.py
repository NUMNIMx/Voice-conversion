import streamlit as st
import soundfile as sf
from pydub import AudioSegment
import os
import torch
import torchaudio
from rvc_model import RVCModel

# ฟังก์ชันแปลง MP3 เป็น WAV
def convert_mp3_to_wav(mp3_file, wav_path):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_path, format="wav")
    return wav_path

# โหลด Hubert model
hubert_model = torchaudio.models.hubert_base()
hubert_model.eval()

# ฟังก์ชัน extract feature ด้วย Hubert
def extract_hubert_feature(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    with torch.no_grad():
        features = hubert_model(waveform)
    return features, sr

# ฟังก์ชันโหลดโมเดล RVC v2
def load_model(model_file):
    model = torch.load(model_file.name, map_location=torch.device('cpu'))
    model.eval()
    return model

# ฟังก์ชันใช้งานโมเดล
def use_model(model, audio_b_path, audio_a_path=None):
    features_b, sr_b = extract_hubert_feature(audio_b_path)
    input_tensor = features_b.mean(dim=1)  # ปรับตาม input ที่ RVC v2 ต้องการ
    with torch.no_grad():
        predicted_output = model(input_tensor)
    generated_audio = predicted_output.numpy()  # ปรับตาม output จริง
    return generated_audio, sr_b

# Streamlit GUI
st.title("Voice Cloning MiniProject with RVC v2")

# ตัวเลือกโหมด
mode = st.radio("เลือกโหมดการทำงาน:", ("Use Existing Model",))  # RVC v2 ไม่ต้องเทรนใหม่ในที่นี้

model = None
if mode == "Use Existing Model":
    model_file = st.file_uploader("อัปโหลดไฟล์โมเดล RVC v2 (.pth)", type=["pth"])
    if model_file:
        model = load_model(model_file)
        st.session_state["model"] = model
        st.success("โหลดโมเดลสำเร็จ!")
    else:
        st.warning("กรุณาอัปโหลดไฟล์โมเดล")

# อัปโหลดไฟล์ B
audio_b_file = st.file_uploader("อัปโหลดไฟล์เสียงของ B (MP3)", type=["mp3"], key="audio_b")
if audio_b_file:
    audio_b_path = convert_mp3_to_wav(audio_b_file, "audio_b.wav")
    st.success("อัปโหลดไฟล์ B สำเร็จ!")

# ปุ่มสร้างผลลัพธ์
if st.button("Generate Cloned Voice"):
    if audio_b_file and "model" in st.session_state:
        generated_audio, sr = use_model(st.session_state["model"], audio_b_path)
        sf.write("generated_audio.wav", generated_audio, sr)
        st.audio("generated_audio.wav")
        st.success("สร้างไฟล์ผลลัพธ์สำเร็จ!")
    else:
        st.error("กรุณาโหลดโมเดลและอัปโหลดไฟล์ B ก่อน")