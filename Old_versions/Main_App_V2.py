import os
import sys
import time
import json
import threading
import shutil
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import librosa
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from fairseq import checkpoint_utils

# กำหนดค่า directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# สร้าง directory หากยังไม่มี
for dir_path in [MODELS_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class RVCV2:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hubert_model = None
        self.current_model = None
        self.model_name = None

    def load_hubert(self):
        """โหลด Hubert model สำหรับการสกัด feature"""
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                ["hubert_base.pt"],
                suffix="",
            )
            hubert_model = models[0]
            hubert_model = hubert_model.to(self.device)
            hubert_model.eval()
            self.hubert_model = hubert_model
            return hubert_model
        except Exception as e:
            print(f"Error loading Hubert model: {e}")
            return None

    def extract_features(self, audio_path, sr=16000):
        """สกัด features จากไฟล์เสียง"""
        if self.hubert_model is None:
            self.load_hubert()
        
        # โหลดและ resample เสียง
        audio, sr_orig = librosa.load(audio_path, sr=None)
        if sr_orig != sr:
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr)
        
        # แปลงเป็น tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        # สกัด features
        with torch.no_grad():
            features = self.hubert_model.extract_features(
                source=audio_tensor,
                padding_mask=None,
                mask=False,
                output_layer=9,
            )[0]
        
        return features.squeeze(0).cpu().numpy()

    def train_model(self, audio_path, model_name, epochs=100, batch_size=4, callback=None):
        """เทรนโมเดล RVC V2"""
        try:
            # สกัด features
            features = self.extract_features(audio_path)
            
            # สร้างโมเดล
            model = torch.nn.Sequential(
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256)
            ).to(self.device)
            
            # กำหนด optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            
            # เทรนโมเดล
            for epoch in range(epochs):
                # จำลอง batch
                for i in range(0, len(features), batch_size):
                    batch = torch.tensor(features[i:i+batch_size]).to(self.device)
                    
                    # Forward pass
                    outputs = model(batch)
                    
                    # คำนวณ loss
                    loss = torch.nn.functional.mse_loss(outputs, batch)
                    
                    # Backward pass และ optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # อัปเดตความคืบหน้า
                progress = (epoch + 1) / epochs * 100
                if callback:
                    callback(progress)
            
            # บันทึกโมเดล
            model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            config_path = os.path.join(MODELS_DIR, f"{model_name}.json")
            
            torch.save(model.state_dict(), model_path)
            
            # บันทึกการตั้งค่า
            config = {
                "model_name": model_name,
                "epochs": epochs,
                "sample_rate": 16000,
                "feature_dim": 768,
                "output_dim": 256,
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            return True, model_path
            
        except Exception as e:
            return False, str(e)

    def load_model(self, model_path):
        """โหลดโมเดล RVC V2"""
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 768),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256)
            ).to(self.device)
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            
            self.current_model = model
            self.model_name = os.path.basename(model_path).replace('.pth', '')
            
            return True, "โหลดโมเดลสำเร็จ"
        except Exception as e:
            return False, f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}"

    def convert_voice(self, input_path, output_path, f0_scale=1.0, callback=None):
        """แปลงเสียงโดยใช้โมเดล RVC V2"""
        if self.current_model is None:
            return False, "ไม่ได้โหลดโมเดล โปรดโหลดโมเดลก่อน"
        
        try:
            # โหลดไฟล์เสียง
            audio, sr = librosa.load(input_path, sr=16000)
            
            # สกัด f0 (fundamental frequency)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # ปรับ f0 ตาม scale
            f0 = f0 * f0_scale
            
            # แบ่งเสียงเป็นเฟรม
            hop_length = 320  # 20ms
            frame_length = 1024  # 64ms
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            
            # สกัด features สำหรับแต่ละเฟรม
            features = []
            total_frames = frames.shape[1]
            
            for i in range(total_frames):
                if callback:
                    progress = (i + 1) / total_frames * 100
                    callback(progress)
                
                frame = frames[:, i]
                # สกัด feature โดยใช้ฟังก์ชัน extract_hubert_frame ที่แก้ไขแล้ว
                feature = self.extract_hubert_frame(frame)
                features.append(feature)
            
            features = np.array(features)
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # แปลงเสียงโดยใช้โมเดล
            with torch.no_grad():
                converted_features = self.current_model(features_tensor).cpu().numpy()
            
            # สังเคราะห์เสียงจาก features และ f0 โดยใช้ synthesize_audio ที่แก้ไขแล้ว
            converted_audio = self.synthesize_audio(converted_features, f0)
            
            # บันทึกไฟล์เสียง
            sf.write(output_path, converted_audio, sr)
            
            return True, output_path
            
        except Exception as e:
            return False, f"เกิดข้อผิดพลาดในการแปลงเสียง: {str(e)}"

    def extract_hubert_frame(self, frame):
        """สกัด feature จากเฟรมเสียงโดยใช้ Hubert (แก้ไขใหม่)"""
        # แปลงเฟรมเป็น tensor
        frame_tensor = torch.FloatTensor(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.hubert_model is None:
                # หากยังไม่มีโมเดล ให้โหลดหรือจำลองค่า
                return np.random.rand(768)
            else:
                features = self.hubert_model.extract_features(
                    source=frame_tensor,
                    padding_mask=None,
                    mask=False,
                    output_layer=9,
                )[0]
                # เอาเฉพาะเฟรมแรก (จำลอง)
                return features.squeeze(0).cpu().numpy()[0]

    def synthesize_audio(self, features, f0):
        """สังเคราะห์เสียงจาก features และ f0 (แก้ไขใหม่)"""
        # กำหนดความยาวของเสียงตามจำนวนเฟรม
        hop_length = 320
        audio_length = features.shape[0] * hop_length
        
        # สร้าง time axis สำหรับเสียง
        time_axis = np.arange(audio_length) / 16000  # sample rate = 16000
        
        # แปลง f0 จาก per-frame เป็น per-sample โดยการ interpolate
        f0_clean = np.nan_to_num(f0, nan=0.0)
        if len(f0_clean) > 1:
            f0_interp = np.interp(
                np.arange(0, len(f0_clean) * hop_length, hop_length/4), 
                np.arange(0, len(f0_clean) * hop_length, hop_length), 
                np.repeat(f0_clean, 4)
            )
        else:
            f0_interp = np.zeros(audio_length)
        f0_interp = f0_interp[:audio_length]
        
        # สร้างเสียงพื้นฐานด้วย sine wave
        base_audio = 0.3 * np.sin(2 * np.pi * np.cumsum(f0_interp) / 16000)
        
        # สร้าง modulation จาก features (จำลอง)
        modulation = np.zeros(audio_length)
        for i, feat in enumerate(features):
            start = i * hop_length
            end = min(start + hop_length, audio_length)
            modulation[start:end] = np.mean(feat) * 0.01
        
        # รวมเสียงพื้นฐานกับ modulation
        synthesized_audio = base_audio + modulation
        
        # Normalize เสียงให้อยู่ในช่วงที่เหมาะสม
        synthesized_audio = synthesized_audio / np.max(np.abs(synthesized_audio)) * 0.9
        
        return synthesized_audio

class RVCGui:
    def __init__(self, root):
        self.root = root
        self.root.title("RVC V2 - Voice Conversion")
        self.root.geometry("800x600")
        
        # สร้าง RVC engine
        self.rvc = RVCV2()
        
        # ตัวแปรสำหรับการบันทึกเสียง
        self.recording = False
        self.sample_rate = 16000
        self.recorded_data = []
        self.last_converted_file = None
        
        self.create_ui()
    
    def create_ui(self):
        # สร้าง notebook สำหรับแท็บ 2 โหมด
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # สร้างแท็บทั้ง 2 โหมด
        train_tab = ttk.Frame(notebook)
        convert_tab = ttk.Frame(notebook)
        
        notebook.add(train_tab, text="โหมดเทรนเสียงเป็นโมเดล")
        notebook.add(convert_tab, text="โหมดการนำโมเดลไปใช้งานกับเสียงผู้อื่น")
        
        # สร้าง UI สำหรับแท็บเทรนเสียง
        self.create_training_ui(train_tab)
        
        # สร้าง UI สำหรับแท็บใช้งานโมเดล
        self.create_conversion_ui(convert_tab)
    
    def create_training_ui(self, parent):
        header_label = ttk.Label(parent, text="เทรนเสียงเป็นโมเดล RVC V2", font=("Arial", 14, "bold"))
        header_label.pack(pady=10)
        
        desc_label = ttk.Label(parent, text="บันทึกเสียงหรือเลือกไฟล์เสียงของคุณเพื่อเทรนเป็นโมเดล", font=("Arial", 10))
        desc_label.pack(pady=5)
        
        # กรอบสำหรับการบันทึกเสียง
        recording_frame = ttk.LabelFrame(parent, text="การบันทึกเสียง")
        recording_frame.pack(fill='x', padx=10, pady=10)
        
        self.record_button = ttk.Button(recording_frame, text="เริ่มบันทึกเสียง", command=self.toggle_recording)
        self.record_button.pack(side='left', padx=10, pady=10)
        
        self.record_status = ttk.Label(recording_frame, text="พร้อมบันทึก")
        self.record_status.pack(side='left', padx=10, pady=10)
        
        # กรอบสำหรับเลือกไฟล์เสียง
        file_frame = ttk.LabelFrame(parent, text="หรือเลือกไฟล์เสียง")
        file_frame.pack(fill='x', padx=10, pady=10)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=60).pack(side='left', padx=10, pady=10)
        ttk.Button(file_frame, text="เลือกไฟล์", command=self.browse_audio_file).pack(side='left', padx=10, pady=10)
        
        # กรอบสำหรับการตั้งค่าโมเดล
        model_frame = ttk.LabelFrame(parent, text="การตั้งค่าโมเดล")
        model_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(model_frame, text="ชื่อโมเดล:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.model_name = tk.StringVar(value="my_voice_model")
        ttk.Entry(model_frame, textvariable=self.model_name, width=30).grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(model_frame, text="จำนวนรอบเทรน:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.epochs = tk.IntVar(value=100)
        ttk.Entry(model_frame, textvariable=self.epochs, width=10).grid(row=1, column=1, padx=10, pady=10, sticky='w')
        
        train_button = ttk.Button(parent, text="เริ่มเทรนโมเดล", command=self.start_training)
        train_button.pack(padx=10, pady=20)
        
        progress_frame = ttk.LabelFrame(parent, text="ความคืบหน้า")
        progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=700, mode="determinate")
        self.progress.pack(padx=10, pady=10)
        
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(padx=10, pady=5)
    
    def create_conversion_ui(self, parent):
        header_label = ttk.Label(parent, text="แปลงเสียงด้วย RVC V2", font=("Arial", 14, "bold"))
        header_label.pack(pady=10)
        
        desc_label = ttk.Label(parent, text="เลือกโมเดลและไฟล์เสียงที่ต้องการแปลง", font=("Arial", 10))
        desc_label.pack(pady=5)
        
        model_frame = ttk.LabelFrame(parent, text="เลือกโมเดล")
        model_frame.pack(fill='x', padx=10, pady=10)
        
        self.model_list = ttk.Combobox(model_frame, width=50)
        self.model_list.pack(side='left', padx=10, pady=10)
        
        refresh_button = ttk.Button(model_frame, text="รีเฟรช", command=self.refresh_models)
        refresh_button.pack(side='left', padx=10, pady=10)
        
        load_button = ttk.Button(model_frame, text="โหลดโมเดล", command=self.load_selected_model)
        load_button.pack(side='left', padx=10, pady=10)
        
        input_frame = ttk.LabelFrame(parent, text="เลือกไฟล์เสียงที่ต้องการแปลง")
        input_frame.pack(fill='x', padx=10, pady=10)
        
        self.input_file_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_file_path, width=60).pack(side='left', padx=10, pady=10)
        ttk.Button(input_frame, text="เลือกไฟล์", command=self.browse_input_file).pack(side='left', padx=10, pady=10)
        
        record_input_button = ttk.Button(input_frame, text="บันทึกเสียงเข้า", command=self.record_input)
        record_input_button.pack(side='left', padx=10, pady=10)
        
        settings_frame = ttk.LabelFrame(parent, text="ตั้งค่าการแปลง")
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(settings_frame, text="ชื่อไฟล์เสียงที่แปลงแล้ว:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.output_name = tk.StringVar(value="converted_voice")
        ttk.Entry(settings_frame, textvariable=self.output_name, width=30).grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(settings_frame, text="ปรับระดับเสียง (F0 Scale):").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.f0_scale = tk.DoubleVar(value=1.0)
        ttk.Scale(settings_frame, from_=0.5, to=2.0, length=200, orient="horizontal", variable=self.f0_scale).grid(row=1, column=1, padx=10, pady=10)
        self.f0_scale_label = ttk.Label(settings_frame, text="1.0")
        self.f0_scale_label.grid(row=1, column=2, padx=10, pady=10)
        self.f0_scale.trace_add("write", self.update_f0_scale_label)
        
        convert_button = ttk.Button(parent, text="แปลงเสียง", command=self.convert_voice)
        convert_button.pack(padx=10, pady=20)
        
        convert_progress_frame = ttk.LabelFrame(parent, text="ความคืบหน้า")
        convert_progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.convert_progress = ttk.Progressbar(convert_progress_frame, orient="horizontal", length=700, mode="determinate")
        self.convert_progress.pack(padx=10, pady=10)
        
        self.convert_progress_label = ttk.Label(convert_progress_frame, text="0%")
        self.convert_progress_label.pack(padx=10, pady=5)
        
        self.model_status = ttk.Label(parent, text="สถานะ: ยังไม่ได้โหลดโมเดล", font=("Arial", 10))
        self.model_status.pack(pady=5)
        
        self.refresh_models()
    
    def update_f0_scale_label(self, *args):
        self.f0_scale_label.config(text=f"{self.f0_scale.get():.2f}")
    
    def refresh_models(self):
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
        self.model_list['values'] = model_files
        if model_files:
            self.model_list.current(0)
    
    def load_selected_model(self):
        if not self.model_list.get():
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกโมเดล")
            return
        
        model_path = os.path.join(MODELS_DIR, self.model_list.get())
        success, message = self.rvc.load_model(model_path)
        
        if success:
            self.model_status.config(text=f"สถานะ: โหลดโมเดล {self.rvc.model_name} แล้ว")
            messagebox.showinfo("สำเร็จ", message)
        else:
            self.model_status.config(text="สถานะ: ยังไม่ได้โหลดโมเดล")
            messagebox.showerror("ข้อผิดพลาด", message)
    
    def browse_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if file_path:
            self.file_path.set(file_path)
    
    def browse_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if file_path:
            self.input_file_path.set(file_path)
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.recording = True
        self.recorded_data = []
        self.record_button.config(text="หยุดบันทึก")
        self.record_status.config(text="กำลังบันทึก...")
        
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()
    
    def record_audio(self):
        def callback(indata, frames, time_info, status):
            self.recorded_data.append(indata.copy())
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
            while self.recording:
                sd.sleep(100)
    
    def stop_recording(self):
        self.recording = False
        self.record_button.config(text="เริ่มบันทึกเสียง")
        self.record_status.config(text="บันทึกเสร็จสิ้น")
        
        if self.recorded_data:
            recorded_audio = np.concatenate(self.recorded_data, axis=0)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")
            write(filename, self.sample_rate, recorded_audio)
            
            self.file_path.set(filename)
            messagebox.showinfo("บันทึกเสียง", f"บันทึกเสียงเสร็จสิ้น: {filename}")
    
    def record_input(self):
        if not self.recording:
            self.start_recording()
            self.record_thread = threading.Thread(target=self.wait_for_recording_complete)
            self.record_thread.daemon = True
            self.record_thread.start()
        else:
            self.stop_recording()
    
    def wait_for_recording_complete(self):
        while self.recording:
            time.sleep(0.1)
        self.input_file_path.set(self.file_path.get())
    
    def start_training(self):
        if not self.file_path.get():
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกไฟล์เสียงหรือบันทึกเสียงก่อน")
            return
        
        model_name = self.model_name.get()
        if not model_name:
            messagebox.showerror("ข้อผิดพลาด", "กรุณาระบุชื่อโมเดล")
            return
        
        train_thread = threading.Thread(target=self.train_model_thread)
        train_thread.daemon = True
        train_thread.start()
    
    def train_model_thread(self):
        try:
            self.root.after(0, lambda: self.update_progress(0))
            success, result = self.rvc.train_model(
                self.file_path.get(),
                self.model_name.get(),
                epochs=self.epochs.get(),
                callback=lambda p: self.root.after(0, lambda: self.update_progress(p))
            )
            if success:
                self.root.after(0, lambda: messagebox.showinfo("สำเร็จ", f"เทรนโมเดลเสร็จสิ้น: {result}"))
                self.root.after(0, self.refresh_models)
            else:
                self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดระหว่างการเทรน: {result}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}"))
    
    def update_progress(self, value):
        self.progress['value'] = value
        self.progress_label.config(text=f"{value:.1f}%")
    
    def convert_voice(self):
        if not self.input_file_path.get():
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกไฟล์เสียงหรือบันทึกเสียงก่อน")
            return
        if self.rvc.current_model is None:
            messagebox.showerror("ข้อผิดพลาด", "กรุณาโหลดโมเดลก่อน")
            return
        
        output_name = self.output_name.get() if self.output_name.get() else "converted_voice"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{output_name}_{timestamp}.wav")
        
        convert_thread = threading.Thread(target=self.convert_voice_thread, args=(output_path,))
        convert_thread.daemon = True
        convert_thread.start()
    
    def convert_voice_thread(self, output_path):
        try:
            self.root.after(0, lambda: self.update_convert_progress(0))
            success, result = self.rvc.convert_voice(
                self.input_file_path.get(),
                output_path,
                f0_scale=self.f0_scale.get(),
                callback=lambda p: self.root.after(0, lambda: self.update_convert_progress(p))
            )
            if success:
                self.last_converted_file = result
                self.root.after(0, lambda: messagebox.showinfo("สำเร็จ", f"แปลงเสียงเสร็จสิ้น: {result}"))
                self.root.after(0, lambda: self.play_converted_audio(result))
            else:
                self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดระหว่างการแปลงเสียง: {result}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}"))
    
    def update_convert_progress(self, value):
        self.convert_progress['value'] = value
        self.convert_progress_label.config(text=f"{value:.1f}%")
    
    def play_converted_audio(self, audio_path):
        try:
            data, samplerate = sf.read(audio_path)
            sd.play(data, samplerate)
            self.create_playback_controls()
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถเล่นไฟล์เสียง: {str(e)}")
    
    def create_playback_controls(self):
        if hasattr(self, 'playback_frame'):
            self.playback_frame.destroy()
        self.playback_frame = ttk.LabelFrame(self.root, text="การเล่นเสียง")
        self.playback_frame.pack(fill='x', padx=10, pady=10)
        stop_button = ttk.Button(self.playback_frame, text="หยุดเล่น", command=self.stop_audio)
        stop_button.pack(side='left', padx=10, pady=10)
        save_button = ttk.Button(self.playback_frame, text="บันทึกไฟล์", command=self.save_converted_audio)
        save_button.pack(side='left', padx=10, pady=10)
    
    def stop_audio(self):
        sd.stop()
    
    def save_converted_audio(self):
        if not self.last_converted_file:
            messagebox.showerror("ข้อผิดพลาด", "ไม่มีไฟล์เสียงที่แปลงล่าสุด")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if save_path:
            try:
                shutil.copy(self.last_converted_file, save_path)
                messagebox.showinfo("สำเร็จ", f"บันทึกไฟล์เสียงไปยัง: {save_path}")
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถบันทึกไฟล์: {str(e)}")
    
    def cleanup(self):
        sd.stop()
        for file in os.listdir(TEMP_DIR):
            try:
                os.remove(os.path.join(TEMP_DIR, file))
            except Exception:
                pass

if __name__ == "__main__":
    root = tk.Tk()
    app = RVCGui(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()
