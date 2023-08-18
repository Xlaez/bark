from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
import scipy

config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="bark/", eval=True)

text = "Subscirbe to my channel for more videos!"
output_dict = model.synthesize(text, config, speaker_id="speaker_one", voice_dirs="bark-voices/")


sample_rate = 24000
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=output_dict["wav"])

# from TTS.tts.configs.bark_config import BarkConfig
# from TTS.tts.models.bark import Bark
# import scipy
# import torch

# config = BarkConfig()
# model = Bark.init_from_config(config)

# try:
#     checkpoint_path = "bark/pytorch_model.bin"  # Update this with the correct path
#     print(f"Loading checkpoint from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Use "cpu" for map_location
#     model.load_state_dict(checkpoint["model"])

#     text = "Subscirbe to my channel for more videos!"
#     output_dict = model.synthesize(text, config, speaker_id="speaker_one", voice_dirs="bark-voices/")

#     sample_rate = 24000  # Correct the sample rate to the appropriate value
#     scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=output_dict["wav"])
# except Exception as e:
#     print(f"An error occurred: {e}")
