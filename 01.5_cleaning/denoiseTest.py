import noisereduce as nr
import soundfile as sf

audio, sr = sf.read("/path/to/user/Desktop/AudioThreeTest/otherAudio/CSA-Clinical/Spliced Audio/Spontaneous Speech/124319-6-8-20-SpontaneousSpeech-11-4-22-EM.wav")


print(f"Audio shape: {audio.shape}, Sample rate: {sr}, Duration: {len(audio)/sr:.2f}s")

cleaned = nr.reduce_noise(
    y=audio,
    sr=sr,
    stationary=True,
    prop_decrease=0.8,
    n_fft=2048,
    hop_length=512,
    win_length=2048
)

print("Success!")
