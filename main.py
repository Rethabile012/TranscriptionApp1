import tkinter as tk
from tkinter import filedialog, messagebox
import requests
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa.display

from extractedmfccs import AudioProcessor
from acoustic_model import BiLSTM
from trainer import TextEncoder
from decoder import CEDecoder

API_BASE = "https://localhost:7191/api"
TOKEN = None  
MODEL_PATH = "acoustic_model_best.npz"

def softmax_rows(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def load_model(model_path=MODEL_PATH, input_size=13, hidden_size=256, output_size=None, lr=0.005):
    encoder = TextEncoder()
    if output_size is None:
        output_size = len(encoder.chars)

    model = BiLSTM(input_size, hidden_size, output_size, lr)
    data = np.load(model_path)

    if hasattr(model, "layers"):
        for i, layer in enumerate(model.layers):
            layer.Wf = data[f"Wf_{i}"]
            layer.Wi = data[f"Wi_{i}"]
            layer.Wc = data[f"Wc_{i}"]
            layer.Wo = data[f"Wo_{i}"]

            layer.bf = data[f"bf_{i}"]
            layer.bi = data[f"bi_{i}"]
            layer.bc = data[f"bc_{i}"]
            layer.bo = data[f"bo_{i}"]

    model.Wy = data["Wy"]
    model.by = data["by"]

    return model, encoder

def predict(audio_file, model, encoder, decoder):
    processor = AudioProcessor(audio_file)
    signal, sr = processor.load_audio()
    mfcc = processor.extract_mfcc(signal, sr)
    inputs = mfcc.T.astype(np.float32)

    outputs = model.forward(inputs)
    y_probs = softmax_rows(outputs)
    return decoder.greedy_decode(y_probs)

class SpeechToTextApp:
    def __init__(self, root, model, encoder, decoder):
        self.root = root
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.root.title("Speech-to-Text GUI")
        self.root.geometry("900x700")
        self.show_register_screen()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def get_audio_duration(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        minutes = duration / 60
        return round(minutes, 2)

    def show_register_screen(self):
        self.clear_screen()
        tk.Label(self.root, text="Register", font=("Arial", 16)).pack(pady=10)

        self.reg_email = tk.Entry(self.root, width=30)
        self.reg_first = tk.Entry(self.root, width=30)
        self.reg_last = tk.Entry(self.root, width=30)
        self.reg_pass = tk.Entry(self.root, show="*", width=30)

        tk.Label(self.root, text="Email").pack()
        self.reg_email.pack()
        tk.Label(self.root, text="First Name").pack()
        self.reg_first.pack()
        tk.Label(self.root, text="Last Name").pack()
        self.reg_last.pack()
        tk.Label(self.root, text="Password").pack()
        self.reg_pass.pack()

        tk.Button(self.root, text="Register", command=self.register_user).pack(pady=10)
        tk.Button(self.root, text="Go to Login", command=self.show_login_screen).pack()

    def register_user(self):
        data = {
            "email": self.reg_email.get(),
            "fname": self.reg_first.get(),
            "lname": self.reg_last.get(),
            "password": self.reg_pass.get()
        }
        try:
            resp = requests.post(f"{API_BASE}/Auth/register", json=data, verify=False)
            if resp.status_code == 200:
                messagebox.showinfo("Success", "Registered successfully!")
                self.show_login_screen()
            else:
                messagebox.showerror("Error", f"Failed: {resp.text}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_login_screen(self):
        self.clear_screen()
        tk.Label(self.root, text="Login", font=("Arial", 16)).pack(pady=10)

        self.login_email = tk.Entry(self.root, width=30)
        self.login_pass = tk.Entry(self.root, show="*", width=30)

        tk.Label(self.root, text="Email").pack()
        self.login_email.pack()
        tk.Label(self.root, text="Password").pack()
        self.login_pass.pack()

        tk.Button(self.root, text="Login", command=self.login_user).pack(pady=10)
        tk.Button(self.root, text="Go to Register", command=self.show_register_screen).pack()

    def login_user(self):
        global TOKEN
        data = {
            "email": self.login_email.get(),
            "password": self.login_pass.get()
        }
        try:
            resp = requests.post(f"{API_BASE}/Auth/login", json=data, verify=False)
            if resp.status_code == 200:
                TOKEN = resp.text.strip()
                messagebox.showinfo("Success", "Login successful!")
                self.show_upload_screen()
            else:
                messagebox.showerror("Error", f"Login failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_audio_plots(self, file_path):
        try:
            signal, sr = librosa.load(file_path, sr=None)

            for frame in [self.waveform_frame, self.spectrogram_frame]:
                for widget in frame.winfo_children():
                    widget.destroy()

            # Waveform
            fig1 = plt.Figure(figsize=(4.5,2.5), dpi=100)
            ax1 = fig1.add_subplot(111)
            ax1.set_title("Waveform")
            librosa.display.waveshow(signal, sr=sr, ax=ax1)
            canvas1 = FigureCanvasTkAgg(fig1, master=self.waveform_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack()

            # Spectrogram
            stft = librosa.stft(signal)
            spect = librosa.amplitude_to_db(abs(stft))

            fig2 = plt.Figure(figsize=(4.5,2.5), dpi=100)
            ax2 = fig2.add_subplot(111)
            ax2.set_title("Spectrogram")
            librosa.display.specshow(spect, sr=sr, x_axis="time", y_axis="hz", ax=ax2)
            canvas2 = FigureCanvasTkAgg(fig2, master=self.spectrogram_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack()

        except Exception as e:
            messagebox.showerror("Error", f"Audio plot failed:\n{e}")

    def show_upload_screen(self):
        self.clear_screen()
        tk.Label(self.root, text="Upload & Transcribe Audio", font=("Arial", 16)).pack(pady=10)

        self.filepath_var = tk.StringVar()
        self.filename_var = tk.StringVar()
        self.transcription_text = tk.Text(self.root, height=10, width=60)

        tk.Entry(self.root, textvariable=self.filepath_var, width=60).pack()
        tk.Button(self.root, text="Browse", command=self.browse_file).pack(pady=5)

        tk.Entry(self.root, textvariable=self.filename_var, width=60).pack()

        tk.Button(self.root, text="Upload & Transcribe", command=self.upload_and_transcribe).pack(pady=10)

        # Plot frames
        self.waveform_frame = tk.Frame(self.root)
        self.waveform_frame.pack(pady=10)

        self.spectrogram_frame = tk.Frame(self.root)
        self.spectrogram_frame.pack(pady=10)

        tk.Label(self.root, text="Transcription").pack(pady=5)
        self.transcription_text.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            self.filepath_var.set(file_path)
            self.filename_var.set(os.path.basename(file_path))
            self.show_audio_plots(file_path)

    def upload_and_transcribe(self):
        if not TOKEN:
            messagebox.showerror("Error", "Please login first!")
            return

        file_path = self.filepath_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file first!")
            return

        duration = self.get_audio_duration(file_path)
        headers = {"Authorization": f"Bearer {TOKEN}"}
        data = {
            "fileName": self.filename_var.get(),
            "filePath": file_path,
            "duration": duration
        }

        try:
            resp = requests.post(f"{API_BASE}/AudioFiles", json=data, headers=headers, verify=False)
            if resp.status_code != 200:
                messagebox.showerror("Error", f"Upload failed: {resp.text}")
                return

            transcription = predict(file_path, self.model, self.encoder, self.decoder)

            self.transcription_text.delete(1.0, tk.END)
            self.transcription_text.insert(tk.END, transcription)
            messagebox.showinfo("Success", "Uploaded metadata & transcribed!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(str(e))

if __name__ == "__main__":
    model, encoder = load_model()
    decoder = CEDecoder(encoder.idx2char, beam_width=5, alpha=0.5)
    root = tk.Tk()
    app = SpeechToTextApp(root, model, encoder, decoder)
    root.mainloop()
