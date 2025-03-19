import os
import sys
import time
import threading
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import torch
import librosa
from scipy.io.wavfile import write

class VoiceConversionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Retrieval-based Voice Conversion")
        self.root.geometry("800x600")
        
        # ตัวแปรต่างๆ
        self.recording = False
        self.sample_rate = 16000
        self.recorded_data = []
        self.output_folder = "outputs"
        self.models_folder = "models"
        
        # สร้างโฟลเดอร์หากยังไม่มี
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.models_folder, exist_ok=True)
        
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
        # กรอบสำหรับการบันทึกเสียง
        recording_frame = ttk.LabelFrame(parent, text="การบันทึกเสียง")
        recording_frame.pack(fill='x', padx=10, pady=10)
        
        # ปุ่มบันทึกเสียง
        self.record_button = ttk.Button(recording_frame, text="เริ่มบันทึกเสียง", command=self.toggle_recording)
        self.record_button.pack(side='left', padx=10, pady=10)
        
        # ป้ายแสดงสถานะการบันทึก
        self.record_status = ttk.Label(recording_frame, text="พร้อมบันทึก")
        self.record_status.pack(side='left', padx=10, pady=10)
        
        # การเลือกไฟล์เสียง
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
        
        # ปุ่มสำหรับเริ่มการเทรน
        train_button = ttk.Button(parent, text="เริ่มเทรนโมเดล", command=self.start_training)
        train_button.pack(padx=10, pady=20)
        
        # แสดงความคืบหน้า
        progress_frame = ttk.LabelFrame(parent, text="ความคืบหน้า")
        progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=700, mode="determinate")
        self.progress.pack(padx=10, pady=10)
        
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(padx=10, pady=5)
    
    def create_conversion_ui(self, parent):
        # กรอบสำหรับการเลือกโมเดล
        model_frame = ttk.LabelFrame(parent, text="เลือกโมเดล")
        model_frame.pack(fill='x', padx=10, pady=10)
        
        self.model_list = ttk.Combobox(model_frame, width=50)
        self.model_list.pack(side='left', padx=10, pady=10)
        
        refresh_button = ttk.Button(model_frame, text="รีเฟรช", command=self.refresh_models)
        refresh_button.pack(side='left', padx=10, pady=10)
        
        # กรอบสำหรับเลือกไฟล์เสียงที่ต้องการแปลง
        input_frame = ttk.LabelFrame(parent, text="เลือกไฟล์เสียงที่ต้องการแปลง")
        input_frame.pack(fill='x', padx=10, pady=10)
        
        self.input_file_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_file_path, width=60).pack(side='left', padx=10, pady=10)
        ttk.Button(input_frame, text="เลือกไฟล์", command=self.browse_input_file).pack(side='left', padx=10, pady=10)
        
        # ปุ่มสำหรับการบันทึกเสียงเข้า
        record_input_button = ttk.Button(input_frame, text="บันทึกเสียงเข้า", command=self.record_input)
        record_input_button.pack(side='left', padx=10, pady=10)
        
        # กรอบสำหรับการแปลงเสียง
        convert_frame = ttk.LabelFrame(parent, text="การแปลงเสียง")
        convert_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(convert_frame, text="ชื่อไฟล์เสียงที่แปลงแล้ว:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.output_name = tk.StringVar(value="converted_voice")
        ttk.Entry(convert_frame, textvariable=self.output_name, width=30).grid(row=0, column=1, padx=10, pady=10)
        
        # ปุ่มสำหรับการแปลงเสียง
        convert_button = ttk.Button(parent, text="แปลงเสียง", command=self.convert_voice)
        convert_button.pack(padx=10, pady=20)
        
        # แสดงความคืบหน้า
        convert_progress_frame = ttk.LabelFrame(parent, text="ความคืบหน้า")
        convert_progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.convert_progress = ttk.Progressbar(convert_progress_frame, orient="horizontal", length=700, mode="determinate")
        self.convert_progress.pack(padx=10, pady=10)
        
        self.convert_progress_label = ttk.Label(convert_progress_frame, text="0%")
        self.convert_progress_label.pack(padx=10, pady=5)
        
        # โหลดรายการโมเดล
        self.refresh_models()
    
    def refresh_models(self):
        """โหลดรายการโมเดลทั้งหมดในโฟลเดอร์"""
        model_files = [f for f in os.listdir(self.models_folder) if f.endswith(".pth")]
        self.model_list['values'] = model_files
        if model_files:
            self.model_list.current(0)
    
    def browse_audio_file(self):
        """เปิดหน้าต่างเลือกไฟล์เสียง"""
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if file_path:
            self.file_path.set(file_path)
    
    def browse_input_file(self):
        """เปิดหน้าต่างเลือกไฟล์เสียงเข้า"""
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if file_path:
            self.input_file_path.set(file_path)
    
    def toggle_recording(self):
        """สลับการบันทึกเสียง (เริ่ม/หยุด)"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """เริ่มการบันทึกเสียง"""
        self.recording = True
        self.recorded_data = []
        self.record_button.config(text="หยุดบันทึก")
        self.record_status.config(text="กำลังบันทึก...")
        
        # เริ่ม thread สำหรับการบันทึกเสียง
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()
    
    def record_audio(self):
        """ฟังก์ชันสำหรับบันทึกเสียง"""
        def callback(indata, frames, time, status):
            self.recorded_data.append(indata.copy())
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
            while self.recording:
                sd.sleep(100)
    
    def stop_recording(self):
        """หยุดการบันทึกเสียง"""
        self.recording = False
        self.record_button.config(text="เริ่มบันทึกเสียง")
        self.record_status.config(text="บันทึกเสร็จสิ้น")
        
        # บันทึกไฟล์เสียง
        if self.recorded_data:
            recorded_audio = np.concatenate(self.recorded_data, axis=0)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_folder, f"recording_{timestamp}.wav")
            write(filename, self.sample_rate, recorded_audio)
            
            # ตั้งค่า path ไฟล์
            self.file_path.set(filename)
            messagebox.showinfo("บันทึกเสียง", f"บันทึกเสียงเสร็จสิ้น: {filename}")
    
    def record_input(self):
        """บันทึกเสียงสำหรับการแปลง"""
        # ใช้ฟังก์ชันเดียวกันกับการบันทึกเสียงสำหรับการเทรน
        if not self.recording:
            self.start_recording()
            # เปลี่ยนแปลงค่า path เมื่อบันทึกเสร็จ
            self.record_thread = threading.Thread(target=self.wait_for_recording_complete)
            self.record_thread.daemon = True
            self.record_thread.start()
        else:
            self.stop_recording()
    
    def wait_for_recording_complete(self):
        """รอจนกว่าการบันทึกเสร็จสิ้น"""
        while self.recording:
            time.sleep(0.1)
        # เปลี่ยน path หลังจากบันทึกเสร็จ
        self.input_file_path.set(self.file_path.get())
    
    def start_training(self):
        """เริ่มการเทรนโมเดล"""
        if not self.file_path.get():
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกไฟล์เสียงหรือบันทึกเสียงก่อน")
            return
        
        model_name = self.model_name.get()
        if not model_name:
            messagebox.showerror("ข้อผิดพลาด", "กรุณาระบุชื่อโมเดล")
            return
        
        # เริ่ม thread สำหรับการเทรน
        train_thread = threading.Thread(target=self.train_model)
        train_thread.daemon = True
        train_thread.start()
    
    def train_model(self):
        """ฟังก์ชันสำหรับการเทรนโมเดล"""
        try:
            # โหลดไฟล์เสียง
            audio_file = self.file_path.get()
            # จำลองขั้นตอนการเทรน
            epochs = self.epochs.get()
            
            for i in range(epochs):
                # จำลองการเทรน
                time.sleep(0.1)  # ลดเวลาให้เร็วขึ้นเพื่อการทดสอบ
                progress = (i + 1) / epochs * 100
                
                # อัปเดตความคืบหน้า
                self.root.after(0, lambda p=progress: self.update_progress(p))
            
            # สร้างโมเดลจำลอง (ในการใช้งานจริงจะต้องสร้างโมเดลจริง)
            model_file = os.path.join(self.models_folder, f"{self.model_name.get()}.pth")
            
            # จำลองการบันทึกโมเดล
            # ในการใช้งานจริงจะต้องประมวลผลและบันทึกโมเดลจริงๆ
            with open(model_file, 'w') as f:
                f.write("This is a dummy model file")
            
            # อัปเดตรายการโมเดล
            self.root.after(0, self.refresh_models)
            
            # แสดงข้อความสำเร็จ
            self.root.after(0, lambda: messagebox.showinfo("สำเร็จ", f"เทรนโมเดลเสร็จสิ้น: {model_file}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดระหว่างการเทรน: {str(e)}"))
    
    def update_progress(self, value):
        """อัปเดตแถบความคืบหน้า"""
        self.progress['value'] = value
        self.progress_label.config(text=f"{value:.1f}%")
    
    def convert_voice(self):
        """แปลงเสียงโดยใช้โมเดลที่เลือก"""
        if not self.input_file_path.get():
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกไฟล์เสียงที่ต้องการแปลง")
            return
        
        if not self.model_list.get():
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกโมเดล")
            return
        
        # เริ่ม thread สำหรับการแปลง
        convert_thread = threading.Thread(target=self.perform_conversion)
        convert_thread.daemon = True
        convert_thread.start()
    
    def perform_conversion(self):
        """ฟังก์ชันสำหรับการแปลงเสียง"""
        try:
            # โหลดไฟล์เสียงและโมเดล
            input_file = self.input_file_path.get()
            model_file = os.path.join(self.models_folder, self.model_list.get())
            
            # จำลองการแปลงเสียง
            for i in range(100):
                # จำลองการประมวลผล
                time.sleep(0.05)  # ลดเวลาให้เร็วขึ้นเพื่อการทดสอบ
                progress = (i + 1) / 100 * 100
                
                # อัปเดตความคืบหน้า
                self.root.after(0, lambda p=progress: self.update_convert_progress(p))
            
            # สร้างไฟล์เสียงเอาต์พุตจำลอง
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_folder, f"{self.output_name.get()}_{timestamp}.wav")
            
            # ในการใช้งานจริงจะต้องแปลงเสียงจริงๆ และบันทึกผลลัพธ์
            # เราจะจำลองโดยการคัดลอกไฟล์เข้า
            if input_file.endswith('.mp3'):
                # แปลง mp3 เป็น wav
                data, sample_rate = librosa.load(input_file, sr=self.sample_rate)
                sf.write(output_file, data, sample_rate)
            else:
                # คัดลอกไฟล์ wav
                data, sample_rate = sf.read(input_file)
                sf.write(output_file, data, sample_rate)
            
            # แสดงข้อความสำเร็จ
            self.root.after(0, lambda: messagebox.showinfo("สำเร็จ", f"แปลงเสียงเสร็จสิ้น: {output_file}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดระหว่างการแปลงเสียง: {str(e)}"))
    
    def update_convert_progress(self, value):
        """อัปเดตแถบความคืบหน้าการแปลง"""
        self.convert_progress['value'] = value
        self.convert_progress_label.config(text=f"{value:.1f}%")

# ฟังก์ชันเมื่อเริ่มโปรแกรม
def main():
    root = tk.Tk()
    app = VoiceConversionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()