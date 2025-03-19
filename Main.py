import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import pyaudio
import wave
import shutil
from datetime import datetime
import pickle
import torch.nn.functional as F
from scipy import signal

# ============== Constants ==============
SAMPLE_RATE = 16000
HOP_LENGTH = 160
WINDOW_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
F0_MIN = 50
F0_MAX = 1100
N_ITER = 32

# ============== Simple Voice Conversion Model ==============
class SVC_Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sampling_rate = SAMPLE_RATE
        self.is_loaded = False
        
    def load_model(self, model_path):
        """Load the voice conversion model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            self.model = torch.jit.load(model_path)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    def mel_to_audio(self, mel, output_path):
        """Convert mel spectrogram to audio using Griffin-Lim"""
        try:
            # ตรวจสอบและแก้ไขค่าที่ไม่ถูกต้อง
            mel = np.nan_to_num(mel)  # แทนค่า NaN ด้วย 0
            mel = np.clip(mel, a_min=-80, a_max=80)  # จำกัดช่วงค่าเพื่อป้องกัน Infinity
            
            # Convert dB to power
            mel = librosa.db_to_power(mel)

            # ตรวจสอบค่าลบและแทนที่ด้วยค่าขั้นต่ำ
            mel = np.clip(mel, a_min=1e-7, a_max=None)  # หลีกเลี่ยงค่า 0
            
            # Invert mel spectrogram using Griffin-Lim
            n_fft = WINDOW_SIZE
            inversed = librosa.feature.inverse.mel_to_audio(
                M=mel,
                sr=self.sampling_rate,
                n_fft=n_fft,
                hop_length=HOP_LENGTH,
                n_iter=32,
                window='hann',
                center=True
            )

            # ตรวจสอบและแก้ไขค่าผิดปกติในสัญญาณเสียง
            inversed = np.nan_to_num(inversed)
            max_val = np.max(np.abs(inversed))
            if max_val > 0:
                inversed = inversed / max_val  # Normalize
            else:
                inversed = np.zeros_like(inversed)

            sf.write(output_path, inversed, self.sampling_rate)
            return True
        except Exception as e:
            print(f"Error inverting mel spectrogram: {str(e)}")
            return False
    def extract_features(self, audio_path):
        """Extract mel spectrogram and f0 from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            
            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_fft=WINDOW_SIZE, 
                hop_length=HOP_LENGTH, 
                n_mels=80, 
                fmin=0, 
                fmax=8000
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = torch.from_numpy(mel).float().to(self.device)
            
            # Compute f0
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=F0_MIN, 
                fmax=F0_MAX, 
                sr=sr, 
                hop_length=HOP_LENGTH
            )
            
            return mel, f0
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None, None
    
    def convert(self, audio_path, output_path, pitch_shift=0):
        """Convert voice using the loaded model"""
        try:
            if not self.is_loaded:
                raise ValueError("Model not loaded")

            # Extract features
            mel, f0 = self.extract_features(audio_path)
            if mel is None or f0 is None:
                raise ValueError("Failed to extract features")

            # Apply pitch shift
            if pitch_shift != 0:
                f0 = f0 * 2 ** (pitch_shift / 12)

            # Prepare input for model
            mel_input = mel.unsqueeze(0)  # Add batch dimension
            f0_input = torch.from_numpy(f0).float().to(self.device)
            f0_input = f0_input.unsqueeze(0)  # Add batch dimension

            # Generate output
            with torch.no_grad():
                mel_output = self.model(mel_input, f0_input)

            # Convert output mel to audio
            mel_output = mel_output.squeeze().cpu().numpy()
            success = self.mel_to_audio(mel_output, output_path)

            return success
        except Exception as e:
            print(f"Error converting voice: {str(e)}")
            return False

# ============== Simple Voice Trainer ==============
class SVC_Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sampling_rate = SAMPLE_RATE
        
    def prepare_data(self, audio_files, output_dir):
        """Prepare data for training"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            processed_files = []
            for file in audio_files:
                # Load audio
                audio, sr = librosa.load(file, sr=self.sampling_rate, mono=True)
                
                # Compute mel spectrogram
                mel = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=sr, 
                    n_fft=WINDOW_SIZE, 
                    hop_length=HOP_LENGTH, 
                    n_mels=80, 
                    fmin=0, 
                    fmax=8000
                )
                mel = librosa.power_to_db(mel, ref=np.max)
                
                # Compute f0
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio, 
                    fmin=F0_MIN, 
                    fmax=F0_MAX, 
                    sr=sr, 
                    hop_length=HOP_LENGTH
                )
                
                # Save processed data
                filename = os.path.basename(file).split('.')[0]
                output_file = os.path.join(output_dir, f"{filename}_processed.npz")
                np.savez(output_file, mel=mel, f0=f0)
                processed_files.append(output_file)
            
            return processed_files
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return []
    
    def train(self, processed_files, model_path, epochs=100, batch_size=8, update_callback=None):
        """Train the voice conversion model"""
        try:
            # Create a simple model
            class SimpleVoiceModel(torch.nn.Module):
                def __init__(self):
                    super(SimpleVoiceModel, self).__init__()
                    self.encoder = torch.nn.Sequential(
                        torch.nn.Conv1d(80, 256, kernel_size=3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
                        torch.nn.ReLU()
                    )
                    self.decoder = torch.nn.Sequential(
                        torch.nn.Conv1d(256 + 1, 256, kernel_size=3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(256, 80, kernel_size=3, padding=1),
                        torch.nn.Tanh()
                    )
                
                def forward(self, mel, f0):
                    # Encode
                    encoded = self.encoder(mel)
                    
                    # Concatenate f0
                    f0 = F.interpolate(f0.unsqueeze(1), size=encoded.size(2))
                    concat = torch.cat([encoded, f0], dim=1)
                    
                    # Decode
                    output = self.decoder(concat)
                    return output
            
            # Create model
            self.model = SimpleVoiceModel().to(self.device)
            
            # Create optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Load data
            dataset = []
            for file in processed_files:
                data = np.load(file)
                mel = torch.from_numpy(data['mel']).float()
                f0 = torch.from_numpy(data['f0']).float()
                dataset.append((mel, f0))
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i+batch_size]
                    
                    # Prepare batch
                    mel_batch = torch.stack([item[0] for item in batch]).to(self.device)
                    f0_batch = torch.stack([item[1] for item in batch]).to(self.device)
                    
                    # Forward
                    output = self.model(mel_batch, f0_batch)
                    
                    # Loss
                    loss = F.mse_loss(output, mel_batch)
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / (len(dataset) / batch_size)
                if update_callback:
                    update_callback(epoch + 1, epochs, avg_loss)
            
            # Save model
            torch.jit.script(self.model).save(model_path)
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

# ============== Voice Conversion App ==============
class VoiceConversionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RVC Voice Conversion")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.model_path = tk.StringVar()
        self.input_audio_path = tk.StringVar()
        self.output_path = tk.StringVar(value="./output")
        self.model_list = []
        self.training_status = tk.StringVar(value="Ready")
        self.conversion_status = tk.StringVar(value="Ready")
        self.pitch_shift = tk.IntVar(value=0)
        self.record_duration = tk.IntVar(value=5)
        
        # Create SVC model and trainer
        self.svc_model = SVC_Model()
        self.svc_trainer = SVC_Trainer()
        
        # Create tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tab frames
        self.training_tab = ttk.Frame(self.notebook)
        self.conversion_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.training_tab, text="Training Mode")
        self.notebook.add(self.conversion_tab, text="Conversion Mode")
        
        # Setup training tab
        self.setup_training_tab()
        
        # Setup conversion tab
        self.setup_conversion_tab()
        
        # Create directories
        self.create_directories()
        
        # Load model list
        self.load_model_list()
        
        # Audio recording setup
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
    
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./output", exist_ok=True)
        os.makedirs("./recordings", exist_ok=True)
        os.makedirs("./temp", exist_ok=True)
    
    def load_model_list(self):
        """Load list of available models"""
        self.model_list = [f for f in os.listdir("./models") if f.endswith('.pth')]
        self.update_model_dropdown()
    
    def update_model_dropdown(self):
        """Update model dropdown in conversion tab"""
        if hasattr(self, 'model_dropdown'):
            self.model_dropdown['values'] = self.model_list
            if self.model_list:
                self.model_dropdown.current(0)
                self.model_path.set(os.path.join("./models", self.model_list[0]))
    
    def setup_training_tab(self):
        """Setup training tab UI"""
        # Training tab configuration
        frame = ttk.LabelFrame(self.training_tab, text="Train Voice Model")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model name input
        ttk.Label(frame, text="Model Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_name_entry = ttk.Entry(frame, width=30)
        self.model_name_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Training audio input
        ttk.Label(frame, text="Training Audio Files:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.training_audio_listbox = tk.Listbox(frame, width=50, height=5)
        self.training_audio_listbox.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Add and remove buttons for training audio
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Add", command=self.add_training_audio).pack(side=tk.TOP, padx=5, pady=2)
        ttk.Button(btn_frame, text="Remove", command=self.remove_training_audio).pack(side=tk.TOP, padx=5, pady=2)
        
        # Recording section
        rec_frame = ttk.LabelFrame(frame, text="Record Voice")
        rec_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        
        ttk.Label(rec_frame, text="Duration (seconds):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(rec_frame, from_=1, to=60, textvariable=self.record_duration, width=5).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Button(rec_frame, text="Start Recording", command=lambda: self.start_recording("train")).grid(row=0, column=2, sticky="w", padx=5, pady=5)
        
        # Training parameters
        param_frame = ttk.LabelFrame(frame, text="Training Parameters")
        param_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        
        ttk.Label(param_frame, text="Epochs:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.epochs_entry = ttk.Spinbox(param_frame, from_=1, to=1000, width=5)
        self.epochs_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.epochs_entry.insert(0, "100")
        
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.batch_size_entry = ttk.Spinbox(param_frame, from_=1, to=32, width=5)
        self.batch_size_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        self.batch_size_entry.insert(0, "8")
        
        # Training status
        status_frame = ttk.LabelFrame(frame, text="Training Status")
        status_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        
        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        ttk.Label(status_frame, textvariable=self.training_status).grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        
        # Start training button
        ttk.Button(frame, text="Start Training", command=self.start_training).grid(row=5, column=1, sticky="e", padx=5, pady=10)
    
    def setup_conversion_tab(self):
        """Setup conversion tab UI"""
        # Conversion tab configuration
        frame = ttk.LabelFrame(self.conversion_tab, text="Voice Conversion")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model selection
        ttk.Label(frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_dropdown = ttk.Combobox(frame, textvariable=self.model_path, width=30)
        self.model_dropdown.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(frame, text="Load Model", command=self.load_model).grid(row=0, column=2, sticky="w", padx=5, pady=5)
        
        # Input audio selection
        ttk.Label(frame, text="Input Audio:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.input_audio_path, width=30).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_input_audio).grid(row=1, column=2, sticky="w", padx=5, pady=5)
        
        # Record audio option
        ttk.Button(frame, text="Record Audio", command=lambda: self.start_recording("convert")).grid(row=1, column=3, sticky="w", padx=5, pady=5)
        
        # Output path selection
        ttk.Label(frame, text="Output Path:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.output_path, width=30).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_output_path).grid(row=2, column=2, sticky="w", padx=5, pady=5)
        
        # Conversion parameters
        param_frame = ttk.LabelFrame(frame, text="Conversion Parameters")
        param_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=5, pady=10)
        
        ttk.Label(param_frame, text="Pitch Shift:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(param_frame, from_=-24, to=24, textvariable=self.pitch_shift, width=5).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Conversion status
        status_frame = ttk.LabelFrame(frame, text="Conversion Status")
        status_frame.grid(row=4, column=0, columnspan=4, sticky="ew", padx=5, pady=10)
        
        self.conversion_progress_bar = ttk.Progressbar(status_frame, orient="horizontal", length=300, mode="determinate")
        self.conversion_progress_bar.grid(row=0, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        
        ttk.Label(status_frame, textvariable=self.conversion_status).grid(row=1, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        
        # Start conversion button
        ttk.Button(frame, text="Start Conversion", command=self.start_conversion).grid(row=5, column=1, sticky="e", padx=5, pady=10)
        
        # Listen to original and converted audio
        listen_frame = ttk.Frame(frame)
        listen_frame.grid(row=6, column=0, columnspan=4, sticky="ew", padx=5, pady=10)
        
        ttk.Button(listen_frame, text="Play Original", command=self.play_original).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(listen_frame, text="Play Converted", command=self.play_converted).pack(side=tk.LEFT, padx=5, pady=5)
    
    def add_training_audio(self):
        """Add training audio files to the list"""
        files = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav;*.mp3;*.ogg")])
        for file in files:
            self.training_audio_listbox.insert(tk.END, file)
    
    def remove_training_audio(self):
        """Remove selected training audio files from the list"""
        selected = self.training_audio_listbox.curselection()
        if selected:
            self.training_audio_listbox.delete(selected)
    
    def start_recording(self, mode):
        """Start recording audio"""
        if self.recording:
            messagebox.showwarning("Recording", "Already recording")
            return
        
        threading.Thread(target=self.record_audio, args=(mode,)).start()
    
    def record_audio(self, mode):
        """Record audio to a file"""
        try:
            self.recording = True
            self.frames = []
            duration = self.record_duration.get()
            
            if mode == "train":
                self.training_status.set(f"Recording for {duration} seconds...")
            else:
                self.conversion_status.set(f"Recording for {duration} seconds...")
            
            # Open audio stream
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Record audio
            for i in range(0, int(SAMPLE_RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                self.frames.append(data)
                
                # Update progress
                if mode == "train":
                    self.progress_bar["value"] = (i + 1) / (SAMPLE_RATE / CHUNK * duration) * 100
                else:
                    self.conversion_progress_bar["value"] = (i + 1) / (SAMPLE_RATE / CHUNK * duration) * 100
                self.root.update()
            
            # Stop recording
            stream.stop_stream()
            stream.close()
            self.recording = False
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join("./recordings", filename)
            
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            if mode == "train":
                self.training_audio_listbox.insert(tk.END, filepath)
                self.training_status.set("Recording completed")
                self.progress_bar["value"] = 0
            else:
                self.input_audio_path.set(filepath)
                self.conversion_status.set("Recording completed")
                self.conversion_progress_bar["value"] = 0
            
            messagebox.showinfo("Recording", f"Recording completed: {filename}")
        except Exception as e:
            messagebox.showerror("Recording Error", str(e))
            self.recording = False
    
    def start_training(self):
        """Start the training process"""
        # Check if model name is provided
        model_name = self.model_name_entry.get()
        if not model_name:
            messagebox.showerror("Error", "Please enter a model name")
            return
        
        # Check if training audio files are provided
        training_files = list(self.training_audio_listbox.get(0, tk.END))
        if not training_files:
            messagebox.showerror("Error", "Please add training audio files")
            return
        
        # Get training parameters
        try:
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid training parameters")
            return
        
        # Start training in a separate thread
        threading.Thread(target=self.train_model, args=(model_name, training_files, epochs, batch_size)).start()
    
    def train_model(self, model_name, training_files, epochs, batch_size):
        """Train the voice conversion model"""
        try:
            self.training_status.set("Preparing training data...")
            self.progress_bar["value"] = 0
            self.root.update()
            
            # Prepare output directory
            temp_dir = "./temp/training_data"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # Prepare data
            self.training_status.set("Processing audio files...")
            processed_files = self.svc_trainer.prepare_data(training_files, temp_dir)
            
            if not processed_files:
                self.training_status.set("Failed to process audio files")
                messagebox.showerror("Training Error", "Failed to process audio files")
                return
            
            # Create output directory for model
            model_dir = "./models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model_name}.pth")
            
            # Train model
            self.training_status.set("Training model...")
            
            def update_progress(epoch, total_epochs, loss):
                self.progress_bar["value"] = epoch / total_epochs * 100
                self.training_status.set(f"Training: Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
                self.root.update()
            
            success = self.svc_trainer.train(processed_files, model_path, epochs, batch_size, update_progress)
            
            if success:
                self.training_status.set(f"Training completed. Model saved as {model_name}.pth")
                self.progress_bar["value"] = 100
                
                # Update model list
                self.load_model_list()
                
                messagebox.showinfo("Training Complete", f"Model {model_name} has been trained successfully!")
            else:
                self.training_status.set("Training failed")
                messagebox.showerror("Training Error", "Failed to train model")
        except Exception as e:
            self.training_status.set(f"Training failed: {str(e)}")
            messagebox.showerror("Training Error", str(e))
    
    def load_model(self):
        """Load the selected model"""
        model_path = self.model_path.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model")
            return
        
        try:
            self.conversion_status.set("Loading model...")
            self.conversion_progress_bar["value"] = 0
            self.root.update()
            
            success = self.svc_model.load_model(model_path)
            
            if success:
                self.conversion_status.set("Model loaded successfully")
                self.conversion_progress_bar["value"] = 100
                messagebox.showinfo("Model Loaded", "Model loaded successfully")
            else:
                self.conversion_status.set("Failed to load model")
                messagebox.showerror("Model Error", "Failed to load model")
        except Exception as e:
            self.conversion_status.set(f"Failed to load model: {str(e)}")
            messagebox.showerror("Model Error", str(e))
    
    def browse_input_audio(self):
        """Browse for input audio file"""
        file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.ogg")])
        if file:
            self.input_audio_path.set(file)
    
    def browse_output_path(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_path.set(directory)
    
    def start_conversion(self):
        """Start the voice conversion process"""
        # Check if model is loaded
        if not self.svc_model.is_loaded:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        # Check if input audio is provided
        input_audio = self.input_audio_path.get()
        if not input_audio:
            messagebox.showerror("Error", "Please select input audio")
            return
        
        # Check if output path is provided
        output_path = self.output_path.get()
        if not output_path:
            messagebox.showerror("Error", "Please select output path")
            return
        
        # Get conversion parameters
        pitch_shift = self.pitch_shift.get()
        
        # Start conversion in a separate thread
        threading.Thread(target=self.convert_voice, args=(input_audio, output_path, pitch_shift)).start()
    
    def convert_voice(self, input_audio, output_path, pitch_shift):
        """Convert the voice using the loaded model"""
        try:
            self.conversion_status.set("Starting conversion...")
            self.conversion_progress_bar["value"] = 0
            self.root.update()
            
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Set output file path
            filename = os.path.basename(input_audio).split('.')[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_path, f"{filename}_converted_{timestamp}.wav")
            
            # Convert voice
            self.conversion_status.set("Converting voice...")
            self.conversion_progress_bar["value"] = 50
            self.root.update()
            
            success = self.svc_model.convert(input_audio, output_file, pitch_shift)
            
            if success:
                self.conversion_status.set(f"Conversion completed. Output saved to: {output_file}")
                self.conversion_progress_bar["value"] = 100
                messagebox.showinfo("Conversion Complete", f"Voice converted successfully!\nOutput file: {output_file}")
            else:
                self.conversion_status.set("Conversion failed")
                messagebox.showerror("Conversion Error", "Failed to convert voice")
        except Exception as e:
            self.conversion_status.set(f"Conversion failed: {str(e)}")
            messagebox.showerror("Conversion Error", str(e))

    def play_original(self):
        """Play the original audio file"""
        input_audio = self.input_audio_path.get()
        if not input_audio:
            messagebox.showerror("Error", "No input audio selected")
            return
        
        threading.Thread(target=self.play_audio, args=(input_audio,)).start()

    def play_converted(self):
        """Play the converted audio file"""
        # Find the latest converted file in the output directory
        output_path = self.output_path.get()
        if not output_path or not os.path.exists(output_path):
            messagebox.showerror("Error", "No output directory or no converted file found")
            return
        
        # Get latest file
        files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith('.wav') and '_converted_' in f]
        if not files:
            messagebox.showerror("Error", "No converted file found")
            return
        
        latest_file = max(files, key=os.path.getctime)
        threading.Thread(target=self.play_audio, args=(latest_file,)).start()

    def play_audio(self, filepath):
        """Play an audio file"""
        try:
            # Use system default audio player
            if sys.platform == 'win32':  # Windows
                os.startfile(filepath)
            elif sys.platform == 'darwin':  # macOS
                os.system(f"open '{filepath}'")
            else:  # Linux
                os.system(f"xdg-open '{filepath}'")
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

# Main function ควรอยู่นอกคลาส
def main():
    # Set style
    try:
        # Try to use a modern style if available
        import ttkthemes
        root = ttkthemes.ThemedTk()
        root.set_theme("arc")
    except:
        # Fall back to default style
        root = tk.Tk()
        style = ttk.Style()
        if sys.platform == "win32":  # Windows
            style.theme_use('winnative')
        elif sys.platform == "darwin":  # macOS
            style.theme_use('aqua')
        else:  # Linux and others
            style.theme_use('clam')
    
    app = VoiceConversionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()