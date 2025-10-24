import tkinter as tk
from tkinter import filedialog, messagebox
import requests
import os
import librosa
import numpy as np


from extractedmfccs import AudioProcessor
from acoustic_model import BiLSTM
from trainer import TextEncoder
from decoder import CEDecoder

API_BASE = "https://localhost:7191/api"
TOKEN = None  
MODEL_PATH = "acoustic_model_best.npz"


def load_model(model_path=MODEL_PATH, input_size=13, hidden_size=256, output_size=None, lr=0.005):
    encoder = TextEncoder()
    if output_size is None:
        output_size = len(encoder.chars)

    model = BiLSTM(input_size, hidden_size, output_size, lr)
    data = np.load(model_path, allow_pickle=True)

    
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
    signal, sr, _ = processor.load_audio()
    mfcc = processor.extract_mfcc(signal, sr)
    inputs = [mfcc[:, t].reshape(-1, 1) for t in range(mfcc.shape[1])]
    outputs = model.forward(inputs)
    y_probs = np.hstack(outputs).T
    return decoder.beam_search(y_probs)


class SpeechToTextApp:
    def __init__(self, root, model, encoder, decoder):
        self.root = root
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.root.title("Speech-to-Text GUI")
        self.root.geometry("500x500")
        self.show_register_screen()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def get_audio_duration(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)  # duration in seconds

        minutes = duration / 60  # convert to minutes (decimal)
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
            "userEmail": self.reg_email.get(),
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
            "userEmail": self.login_email.get(),
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

    
    def show_upload_screen(self):
        self.clear_screen()
        tk.Label(self.root, text="Upload and Transcribe Audio", font=("Arial", 16)).pack(pady=10)

        self.filepath_var = tk.StringVar()
        self.filename_var = tk.StringVar()
        self.transcription_text = tk.Text(self.root, height=10, width=50)

        tk.Entry(self.root, textvariable=self.filepath_var, width=40).pack()
        tk.Button(self.root, text="Browse", command=self.browse_file).pack(pady=5)

        tk.Entry(self.root, textvariable=self.filename_var, width=40).pack()

        tk.Button(self.root, text="Upload & Transcribe", command=self.upload_and_transcribe).pack(pady=10)

        tk.Label(self.root, text="Transcription").pack(pady=5)
        self.transcription_text.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            self.filepath_var.set(file_path)
            self.filename_var.set(os.path.basename(file_path))
            

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
            # Upload metadata
            resp = requests.post(f"{API_BASE}/AudioFiles", json=data, headers=headers, verify=False)
            if resp.status_code != 200:
                messagebox.showerror("Error", f"Upload failed: {resp.text}")
                return

            # Transcribe locally
            transcription = predict(file_path, self.model, self.encoder, self.decoder)

            # Show transcription in GUI
            self.transcription_text.delete(1.0, tk.END)
            self.transcription_text.insert(tk.END, transcription)
            messagebox.showinfo("Success", f"Uploaded metadata & transcribed!")

        except Exception as e:
            messagebox.showerror("Error", str(e)) 
            print(str(e))



if __name__ == "__main__":
    model, encoder = load_model()
    decoder = CEDecoder(encoder.idx2char, beam_width=5, alpha=0.5)
    root = tk.Tk()
    app = SpeechToTextApp(root, model, encoder, decoder)
    root.mainloop()
