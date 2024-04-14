import os
from pydub import AudioSegment

def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")

# Sử dụng hàm
# convert_m4a_to_wav("path_to_your_file.m4a", "path_to_output_file.wav")


for sub_data_dir in os.listdir("dataset"):
    for index, file in enumerate(os.listdir("dataset/" + sub_data_dir)):
        m4a_path = os.path.join("dataset/" + sub_data_dir, file)
        wav_path = os.path.join("dataset/" + sub_data_dir, f"voice{index}.wav")
        convert_m4a_to_wav(m4a_path, wav_path)