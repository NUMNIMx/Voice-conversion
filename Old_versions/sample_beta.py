import streamlit as st
import soundfile as sf
from pydub import AudioSegment
import os
import torch
import torchaudio
import numpy as np

# สมมติว่ามีคลาส RVCModel (ต้อง implement จริงตาม RVC v2)
class RVCModel:
    def __init__(self):
        # โหลด Hubert model สำหรับ feature extraction
        self.hubert = torchaudio.models.hubert_base()
        self.hubert.eval()
        # สมมติโมเดล RVC (ในที่นี้ใช้โครงสร้างง่ายๆ)
        self.model = torch.nn.Linear(768, 768)  # ปรับตามโมเดลจริง
        
    def train(self, features, sample_rate, epochs=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.model(features.mean(dim=1))
            loss = torch.mean((output - features.mean(dim=1)) ** 2)  # Dummy loss
            loss.backward()
            optimizer.step()
        return self.model.state_dict()

    def convert(self, features):
        with torch.no_grad():
            output = self.model(features.mean(dim=1))
        return output.numpy()  # ปรับตาม output จริง

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

# ฟังก์ชันแปลง MP3 เป็น WAV
def convert_mp3_to_wav(mp3_file, wav_path):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_path, format="wav")
    return wav_path

# ฟังก์ชัน extract feature ด้วย Hubert
def extract_hubert_feature(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    with torch.no_grad():
        outputs = hubert_model(waveform)  # hubert_model มาจาก torchaudio
        features = outputs[0]  # เลือก tensor ที่ต้องการ (ปรับ index ตามเอกสารของ Hubert)
    return features, sr

# โหลด Hubert model
hubert_model = torchaudio.models.hubert_base()
hubert_model.eval()

# Streamlit GUI
st.title("Voice Cloning MiniProject with RVC v2")

# ตัวเลือกโหมด
mode = st.radio("เลือกโหมดการทำงาน:", ("Train New Model", "Use Existing Model"))

if mode == "Train New Model":
    st.header("Train New Model")
    audio_a_file = st.file_uploader("อัปโหลดไฟล์เสียงของ A (MP3 หรือ WAV)", type=["mp3", "wav"], key="audio_a")
    
    if audio_a_file:
        # จัดการไฟล์ที่อัปโหลด
        if audio_a_file.type == "audio/mpeg":
            audio_a_path = convert_mp3_to_wav(audio_a_file, "audio_a.wav")
        else:
            audio_a_path = "audio_a.wav"
            with open(audio_a_path, "wb") as f:
                f.write(audio_a_file.getbuffer())
        st.success("อัปโหลดไฟล์ A สำเร็จ!")
        
        # ปุ่มเทรนโมเดล
        if st.button("เทรนโมเดล"):
            features_a, sr_a = extract_hubert_feature(audio_a_path)
            model = RVCModel()
            state_dict = model.train(features_a, sr_a)
            model_path = "trained_model.pth"
            torch.save(state_dict, model_path)
            st.success("เทรนโมเดลสำเร็จ!")
            st.download_button("ดาวน์โหลดโมเดล", data=open(model_path, "rb"), file_name="trained_model.pth")

elif mode == "Use Existing Model":
    st.header("Use Existing Model")
    
    # อัปโหลดโมเดล
    model_file = st.file_uploader("อัปโหลดไฟล์โมเดล RVC v2 (.pth)", type=["pth"])
    if model_file:
        model_path = "uploaded_model.pth"
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        model = RVCModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        st.session_state["model"] = model
        st.success("โหลดโมเดลสำเร็จ!")
    
    # อัปโหลดไฟล์ B
    audio_b_file = st.file_uploader("อัปโหลดไฟล์เสียงของ B (MP3 หรือ WAV)", type=["mp3", "wav"], key="audio_b")
    if audio_b_file:
        if audio_b_file.type == "audio/mpeg":
            audio_b_path = convert_mp3_to_wav(audio_b_file, "audio_b.wav")
        else:
            audio_b_path = "audio_b.wav"
            with open(audio_b_path, "wb") as f:
                f.write(audio_b_file.getbuffer())
        st.success("อัปโหลดไฟล์ B สำเร็จ!")
        
        # ปุ่มแปลงเสียง
        if st.button("แปลงเสียง"):
            if "model" in st.session_state:
                features_b, sr_b = extract_hubert_feature(audio_b_path)
                generated_audio = st.session_state["model"].convert(features_b)
                output_path = "generated_audio.wav"
                sf.write(output_path, generated_audio, sr_b)
                st.audio(output_path)
                st.success("สร้างไฟล์ผลลัพธ์สำเร็จ!")
            else:
                st.error("กรุณาโหลดโมเดลก่อน")