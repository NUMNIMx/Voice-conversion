import streamlit as st
import librosa
import soundfile as sf
from pydub import AudioSegment
import os
import torch
import torch.nn as nn

# ฟังก์ชันแปลง MP3 เป็น WAV
def convert_mp3_to_wav(mp3_file, wav_path):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_path, format="wav")
    return wav_path

# ตัวอย่างโมเดล PyTorch (placeholder)
class SimpleVoiceModel(nn.Module):
    def __init__(self):
        super(SimpleVoiceModel, self).__init__()
        self.fc1 = nn.Linear(13, 128)  # อินพุต 13, เอาต์พุต 128
        self.fc2 = nn.Linear(128, 64)  # อินพุต 128, เอาต์พุต 64 (แก้จาก 32 เป็น 64)
        self.fc3 = nn.Linear(64, 32)   # อินพุต 64, เอาต์พุต 32 (เพิ่มชั้นนี้)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ใช้ ReLU เป็น activation
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)              # ไม่ต้องมี activation ในชั้นสุดท้าย (ขึ้นกับโมเดลเดิม)
        return x

# ฟังก์ชันเทรนโมเดล (placeholder)
def train_model(audio_a_path):
    audio_a, sr = librosa.load(audio_a_path, sr=None)
    # สมมติว่าเทรนโมเดล (ที่นี่เป็นตัวอย่างง่าย ๆ)
    model = SimpleVoiceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # ตัวอย่างการเทรน (เพิ่มโค้ดจริงตามโมเดลของคุณ)
    # บันทึกโมเดลเป็น .pth
    torch.save(model.state_dict(), "trained_model.pth")
    return model

# ฟังก์ชันโหลดโมเดลจากไฟล์ .pth
def load_model(model_file):
    model = SimpleVoiceModel()  # สร้างโมเดลที่มีโครงสร้างถูกต้อง
    model.load_state_dict(torch.load(model_file.name, map_location=torch.device('cpu')))
    model.eval()  # ตั้งค่าเป็น evaluation mode
    return model

# ฟังก์ชันใช้งานโมเดล (placeholder)
def use_model(model, audio_b_path, audio_a_path=None):
    audio_b, sr_b = librosa.load(audio_b_path, sr=None)
    mfcc_b = librosa.feature.mfcc(y=audio_b, sr=sr_b, n_mfcc=13).T  # (n_frames, 13)
    input_tensor = torch.tensor(mfcc_b, dtype=torch.float32)  # (n_frames, 13)
    with torch.no_grad():
        output_tensor = model(input_tensor)  # (n_frames, 32)
    # Placeholder: แปลง output_tensor กลับเป็น waveform
    generated_audio = audio_b  # ปรับตามจริง
    return generated_audio, sr_b

# Streamlit GUI
st.title("Voice Cloning MiniProject")

# ตัวเลือกโหมด
mode = st.radio("เลือกโหมดการทำงาน:", ("Train Model", "Use Existing Model"))

# ตัวแปรเก็บโมเดล
model = None

# โหมด Train Model
if mode == "Train Model":
    audio_a_file = st.file_uploader("อัปโหลดไฟล์เสียงของ A (MP3)", type=["mp3"], key="train_a")
    if audio_a_file:
        audio_a_path = convert_mp3_to_wav(audio_a_file, "audio_a.wav")
        st.success("อัปโหลดไฟล์ A สำเร็จ!")
    
    if st.button("Train Model"):
        if audio_a_file:
            model = train_model(audio_a_path)
            st.session_state["model"] = model
            st.success("เทรนโมเดลสำเร็จ! โมเดลถูกบันทึกเป็น 'trained_model.pth'")
        else:
            st.error("กรุณาอัปโหลดไฟล์ A ก่อน")

# โหมด Use Existing Model
elif mode == "Use Existing Model":
    model_file = st.file_uploader("อัปโหลดไฟล์โมเดล (.pth)", type=["pth"])
    if model_file:
        model = load_model(model_file)
        st.session_state["model"] = model
        st.success("โหลดโมเดลสำเร็จ!")
    else:
        st.warning("กรุณาอัปโหลดไฟล์โมเดล")

# อัปโหลดไฟล์ B (ใช้ได้ทั้งสองโหมด)
audio_b_file = st.file_uploader("อัปโหลดไฟล์เสียงของ B (MP3)", type=["mp3"], key="audio_b")
if audio_b_file:
    audio_b_path = convert_mp3_to_wav(audio_b_file, "audio_b.wav")
    st.success("อัปโหลดไฟล์ B สำเร็จ!")

# ปุ่มสร้างผลลัพธ์
if st.button("Generate Cloned Voice"):
    if audio_b_file and "model" in st.session_state:
        if mode == "Train Model":
            if not audio_a_file:
                st.error("กรุณาอัปโหลดไฟล์ A เพื่อใช้ในโหมด Train Model")
            else:
                generated_audio, sr = use_model(st.session_state["model"], audio_b_path, audio_a_path)
                sf.write("generated_audio.wav", generated_audio, sr)
                st.audio("generated_audio.wav")
                st.success("สร้างไฟล์ผลลัพธ์สำเร็จ!")
        else:  # Use Existing Model
            generated_audio, sr = use_model(st.session_state["model"], audio_b_path)
            sf.write("generated_audio.wav", generated_audio, sr)
            st.audio("generated_audio.wav")
            st.success("สร้างไฟล์ผลลัพธ์สำเร็จ!")
    else:
        st.error("กรุณาโหลดโมเดลและอัปโหลดไฟล์ B ก่อน")