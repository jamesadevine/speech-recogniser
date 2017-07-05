import wave
import numpy as np

source_wav = wave.open("./hello-world/helloworld.wav")

source_wav_len = source_wav.getnframes()
source_wav_sample = source_wav.getframerate()

source_frames = np.array(source_wav.readframes(source_wav_len))


merge_wavs = [file for file in glob.glob('./noise-wavs/**/*.wav', recursive=True)]

for noise_wav_loc in merge_wavs:
    noise_wav = wave.open(noise_wav_loc)
    noise_wav_len = wave.open(noise_wav_loc)
    noise_wav_sample = wave.open(noise_wav_loc)

    if source_wav_sample != noise_wav_sample:
        print("Cannot process: ", noise_wav_loc," sample rates differ.")
        continue

    noise_frames = np.array(noise_wav.readframes(source_wav_len))

    merged_wav = source_frames / noise_frames

    output_wav = wave.open('helloworld-' + noise_wav_loc[:3] + ".wav")
    output_wav.setframerate(source_wav_sample)
    output_wav.writeframes(merged_wav)

    print("Merged ",noise_wav_loc," with hello world")
