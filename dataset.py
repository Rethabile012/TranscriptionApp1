import os
from extractedmfccs import AudioProcessor  

class Dataset:
    def __init__(self, audio_root="/content/transcriptions/Pipeline 2/TrainingSet/audio", transcript_root="/content/transcriptions/Pipeline 2/TrainingSet/transcripts"):
        self.audio_root = audio_root
        self.transcript_root = transcript_root
        self.processor = AudioProcessor(audio_root)

    def load_text(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Failed to load transcript {filepath}: {e}")
            return ""

    def get_all_data(self):
        dataset = []

        # Walk through audio_root to find all wav files
        for dirpath, _, filenames in os.walk(self.audio_root):
            # Mirror path to transcripts
            relative_folder = os.path.relpath(dirpath, self.audio_root)
            transcript_dirpath = os.path.join(self.transcript_root, relative_folder)

            for filename in filenames:
                if filename.lower().endswith(".wav"):
                    audio_path = os.path.join(dirpath, filename)
                    transcript_path = os.path.join(transcript_dirpath, filename.replace(".wav", ".txt"))

                    if not os.path.exists(transcript_path):
                        print(f"Warning: Missing transcript for {audio_path}")
                        continue

                    try:
                        # Extract MFCCs
                        signal, sr = self.processor.load_audio(audio_path)
                        mfcc = self.processor.extract_mfcc(signal, sr)

                        # Load transcript
                        transcript_text = self.load_text(transcript_path)

                        dataset.append((mfcc, transcript_text))
                        print(f"Loaded pair: {filename} -> {len(transcript_text.split())} characters")
                    except Exception as e:
                        print(f"Failed to process {audio_path}: {e}")

        return dataset

    def get_validation_data(self, audio_root="/content/transcriptions/Pipeline 2/ValidationSet/audio", transcript_root="/content/transcriptions/Pipeline 2/ValidationSet/transcripts"):
        val_processor = AudioProcessor(audio_root)
        validation_data = []

        for dirpath, _, filenames in os.walk(audio_root):
            relative_folder = os.path.relpath(dirpath, audio_root)
            transcript_dirpath = os.path.join(transcript_root, relative_folder)

            for filename in filenames:
                if filename.lower().endswith(".wav"):
                    audio_path = os.path.join(dirpath, filename)
                    transcript_path = os.path.join(transcript_dirpath, filename.replace(".wav", ".txt"))

                    if not os.path.exists(transcript_path):
                        print(f"Warning: Missing transcript for {audio_path}")
                        continue

                    try:
                        signal, sr = val_processor.load_audio(audio_path)
                        mfcc = val_processor.extract_mfcc(signal, sr)
                        transcript_text = self.load_text(transcript_path)

                        validation_data.append((mfcc, transcript_text))
                        print(f"Loaded validation pair: {filename} -> {len(transcript_text.split())} characters")
                    except Exception as e:
                        print(f"Failed to process {audio_path}: {e}")

        return validation_data
    
    """def get_test_data(self, audio_root="/kaggle/input/transcriptions/Pipeline 2/TestSet/audio", transcript_root="/kaggle/input/transcriptions/Pipeline 2/TestSet/transcripts"):
        test_processor = AudioProcessor(audio_root)
        test_data = []

        for dirpath, _, filenames in os.walk(audio_root):
            relative_folder = os.path.relpath(dirpath, audio_root)
            transcript_dirpath = os.path.join(transcript_root, relative_folder)

            for filename in filenames:
                if filename.lower().endswith(".wav"):
                    audio_path = os.path.join(dirpath, filename)
                    transcript_path = os.path.join(transcript_dirpath, filename.replace(".wav", ".txt"))

                    if not os.path.exists(transcript_path):
                        print(f"Warning: Missing transcript for {audio_path}")
                        continue

                    try:
                        signal, sr = test_processor.load_audio(audio_path)
                        mfcc = test_processor.extract_mfcc(signal, sr)
                        transcript_text = self.load_text(transcript_path)

                        test_data.append((mfcc, transcript_text))
                        print(f"Loaded test pair: {filename} -> {len(transcript_text.split())} characters")
                    except Exception as e:
                        print(f"Failed to process {audio_path}: {e}")

        return test_data"""

def main():
    ds = Dataset()
    data = ds.get_all_data()

    print("\nDataset Summary")
    print(f"Total samples loaded: {len(data)}")

    if len(data) > 0:
        mfcc, transcript = data[0]
        print(f"First sample MFCC shape: {mfcc.shape}")
        print(f"First sample transcript: {transcript}")


if __name__ == "__main__":
    main()
