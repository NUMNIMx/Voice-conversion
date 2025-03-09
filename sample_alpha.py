import streamlit as st
import soundfile as sf
from pydub import AudioSegment
import os
import torch
import torchaudio
import torch.nn as nn
import zipfile
import shutil

# ตั้งค่า CSS เพื่อความสวยงาม
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stSuccess {
        background-color: #dff0d8;
        color: #3c763d;
    }
    .stWarning {
        background-color: #fcf8e3;
        color: #8a6d3b;
    }
    .stError {
        background-color: #f2dede;
        color: #a94442;
    }
    </style>
""", unsafe_allow_html=True)

# ฟังก์ชันแปลง MP3 เป็น WAV
def convert_mp3_to_wav(mp3_file, wav_path):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_path, format="wav")
    return wav_path

# โหลด Hubert model
hubert_model = torchaudio.models.hubert_base()
hubert_model.eval()

# นิยาม RVCModel
class RVCModel(nn.Module):
    def __init__(self):
        super(RVCModel, self).__init__()
        self.linear = nn.Linear(768, 80)  # Hubert feature (768) -> mel bins (80)

    def forward(self, x):
        return self.linear(x)  # [seq_len, 768] -> [seq_len, 80]

# ฟังก์ชันเทรนโมเดล
def train_model(dataset_zip_path, epochs=1):
    temp_dir = "temp_dataset"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    source_dir = os.path.join(temp_dir, 'source')
    target_dir = os.path.join(temp_dir, 'target')
    source_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.wav')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.wav')])

    if not source_files or not target_files or len(source_files) != len(target_files):
        raise ValueError("ไฟล์ใน source และ target ไม่ตรงกันหรือไม่มีไฟล์ .wav")

    model = RVCModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        for i, (src_file, tgt_file) in enumerate(zip(source_files, target_files)):
            src_path = os.path.join(source_dir, src_file)
            tgt_path = os.path.join(target_dir, tgt_file)

            src_waveform, src_sr = torchaudio.load(src_path)
            tgt_waveform, tgt_sr = torchaudio.load(tgt_path)

            if src_sr != 16000:
                src_waveform = torchaudio.functional.resample(src_waveform, src_sr, 16000)
            if tgt_sr != 16000:
                tgt_waveform = torchaudio.functional.resample(tgt_waveform, tgt_sr, 16000)

            with torch.no_grad():
                src_features = hubert_model(src_waveform).last_hidden_state[0]  # [seq_len, 768]

            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
            )
            tgt_mel = mel_transform(tgt_waveform).squeeze(0).T  # [num_frames, 80]

            min_len = min(src_features.shape[0], tgt_mel.shape[0])
            src_features = src_features[:min_len]
            tgt_mel = tgt_mel[:min_len]

            optimizer.zero_grad()
            predicted_mel = model(src_features)
            loss = criterion(predicted_mel, tgt_mel)
            loss.backward()
            optimizer.step()

            progress_bar.progress((i + 1) / len(source_files))

        st.write(f"Epoch {epoch+1}/{epochs} เสร็จสิ้น")

    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    shutil.rmtree(temp_dir)  # ล้าง temporary files
    return model_path

# ฟังก์ชันใช้งานโมเดล
def use_model(model, source_path):
    source_waveform, sr = torchaudio.load(source_path)
    if sr != 16000:
        source_waveform = torchaudio.functional.resample(source_waveform, sr, 16000)
        sr = 16000

    with torch.no_grad():
        source_features = hubert_model(source_waveform).last_hidden_state[0]  # [seq_len, 768]
        predicted_mel = model(source_features)  # [seq_len, 80]

    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=320, n_iter=30)
    generated_waveform = griffin_lim(predicted_mel.T)  # [time]
    return generated_waveform.numpy(), sr

# Streamlit GUI
st.title("🎙️ Voice Cloning with RVC v2")
st.markdown("โปรแกรมนี้ช่วยคุณเทรนและใช้งานโมเดลการแปลงเสียงได้อย่างง่ายดาย!")

# ตัวเลือกโหมด
mode = st.radio("เลือกโหมดการทำงาน:", ("Train Model", "Use Model"), horizontal=True)

if mode == "Train Model":
    st.subheader("🛠️ โหมดเทรนโมเดล")
    with st.expander("คำแนะนำ"):
        st.write("""
            - อัปโหลดไฟล์ `.zip` ที่มีโฟลเดอร์ `source` และ `target`
            - ไฟล์ใน `source` และ `target` ต้องเป็น `.wav` และมีชื่อเหมือนกัน
            - การเทรนอาจใช้เวลานาน ขึ้นอยู่กับขนาด dataset และจำนวน epoch
        """)

    col1, col2 = st.columns([2, 1])
    with col1:
        dataset_zip = st.file_uploader("อัปโหลด dataset (.zip)", type=["zip"])
    with col2:
        epochs = st.number_input("จำนวน epochs", min_value=1, value=1, step=1)

    if st.button("เริ่มการเทรน"):
        if dataset_zip:
            with open("dataset.zip", "wb") as f:
                f.write(dataset_zip.read())
            try:
                model_path = train_model("dataset.zip", epochs=epochs)
                st.success("เทรนโมเดลสำเร็จ!")
                with open(model_path, "rb") as f:
                    st.download_button("ดาวน์โหลดโมเดล (.pth)", f, file_name="trained_model.pth")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
        else:
            st.error("กรุณาอัปโหลด dataset")

elif mode == "Use Model":
    st.subheader("🎵 โหมดใช้งานโมเดล")
    with st.expander("คำแนะนำ"):
        st.write("""
            - อัปโหลดไฟล์โมเดล `.pth` ที่เทรนแล้ว
            - อัปโหลดไฟล์ `source audio` (`.mp3` หรือ `.wav`)
            - กดปุ่มเพื่อสร้างเสียงที่แปลงแล้ว
        """)

    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("อัปโหลดโมเดล (.pth)", type=["pth"])
    with col2:
        source_audio = st.file_uploader("อัปโหลด source audio", type=["mp3", "wav"])

    if model_file and source_audio:
        with open("model.pth", "wb") as f:
            f.write(model_file.read())
        model = RVCModel()
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
        model.eval()

        if source_audio.name.endswith('.mp3'):
            source_path = convert_mp3_to_wav(source_audio, "source.wav")
        else:
            with open("source.wav", "wb") as f:
                f.write(source_audio.read())
            source_path = "source.wav"

        if st.button("สร้างเสียงที่แปลงแล้ว"):
            try:
                generated_audio, sr = use_model(model, source_path)
                sf.write("generated_audio.wav", generated_audio, sr)
                st.audio("generated_audio.wav")
                st.success("สร้างไฟล์ผลลัพธ์สำเร็จ!")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
    else:
        st.warning("กรุณาอัปโหลดโมเดลและ source audio")