from transformers import AutoProcessor, BarkModel
import scipy
import os

# os.environ["SUNO_OFFLOAD_CPU"] =  "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")

voice_preset = "v2/en_speaker_6"

inputs = processor("Damn! bro got me unaware [laughs]. I did'nt even see him [laughs]", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

print(audio_array)

#Save the .wav file into the disk
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("got_away.wav", rate=sample_rate, data=audio_array)