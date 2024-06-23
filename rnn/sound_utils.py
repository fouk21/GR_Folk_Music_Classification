import librosa
import numpy as np

class SoundUtils:

    def __init__(self) -> None:
        pass

    @staticmethod
    # Function to segment the audio into fixed-length frames, padding or truncating as necessary
    def segment_audio(audio, frame_length):
        frames = []
        num_frames = len(audio) // frame_length
        
        for i in range(num_frames):
            frame = audio[i * frame_length:(i + 1) * frame_length]
            frames.append(frame)
        
        # Handle the remaining audio
        if len(audio) % frame_length != 0:
            last_frame = audio[num_frames * frame_length:]
            # Pad the last frame with zeros if it's shorter than frame_length
            last_frame = np.pad(last_frame, (0, frame_length - len(last_frame)), 'constant')
            frames.append(last_frame)

        return frames
    
    @staticmethod
    # Function to load and preprocess a WAV file, ensuring fixed length
    def load_wav(file_path, frame_length, fixed_length=None):
        audio, _ = librosa.load(file_path, sr=None)
        if fixed_length:
            if len(audio) < fixed_length:
                # Pad the audio with zeros if it's shorter than fixed_length
                audio = np.pad(audio, (0, fixed_length - len(audio)), 'constant')
            else:
                # Truncate the audio if it's longer than fixed_length
                audio = audio[:fixed_length]
            #audio = (audio - np.mean(audio)) / np.std(audio)
            if frame_length != 0:
                audio_frames = SoundUtils.segment_audio(audio, frame_length)
                return np.array(audio_frames)
            else:
                audio = (audio - np.mean(audio)) / np.std(audio)
                return np.array(audio)
        else:
            return audio