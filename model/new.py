
# # import os
# # os.environ["SOUNDDEVICE_BACKEND"] = "pulseaudio"
# # import time
# # import torch
# # import sounddevice as sd
# # from TTS.tts.configs.xtts_config import XttsConfig
# # from TTS.tts.models.xtts import Xtts

# # def main():
# #     print("Loading model...")

# #     # Load config
# #     config = XttsConfig()
# #     config.load_json("/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained/config.json")

# #     # Init model
# #     model = Xtts.init_from_config(config)
# #     model.load_checkpoint(
# #         config,
# #         checkpoint_dir="xtts_pretrained",
# #         use_deepspeed=False
# #     )
# #     model.cuda()

# #     print("Computing speaker latents...")
# #     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
# #         audio_path=["/home/rajan/Prabal_procit_works/XTTS streaming/00020.wav"]
# #     )

# #     print("Starting real-time inference & playback...")
# #     t0 = time.time()
# #     chunks = model.inference_stream(
# #         "Het heeft me veel tijd gekost om een stem te ontwikkelen, en nu ik hem heb, ga ik niet stil blijven.",
# #         "nl",
# #         gpt_cond_latent,
# #         speaker_embedding
# #     )

# #     for i, chunk in enumerate(chunks):
# #         if i == 0:
# #             print(f"Time to first chunk: {time.time() - t0:.2f} sec")
# #         print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

# #         # Play the chunk immediately at 24 kHz
# #         audio_np = chunk.squeeze().cpu().numpy()
# #         sd.play(audio_np, samplerate=24000, blocking=True)

# #     print("Finished real-time playback.")

# # if __name__ == "__main__":
# #     main()

# #stuttering one
# # import os
# # import time
# # import torch
# # import torchaudio
# # from TTS.tts.configs.xtts_config import XttsConfig
# # from TTS.tts.models.xtts import Xtts
# # import subprocess

# # def main():
# #     print("Loading model...")

# #     # Load config
# #     config = XttsConfig()
# #     config.load_json("/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained/config.json")

# #     # Init model
# #     model = Xtts.init_from_config(config)
# #     model.load_checkpoint(
# #         config,
# #         checkpoint_dir="xtts_pretrained",
# #         use_deepspeed=False
# #     )
# #     model.cuda()

# #     print("Computing speaker latents...")
# #     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
# #         audio_path=["/home/rajan/Prabal_procit_works/XTTS streaming/00020.wav"]
# #     )

# #     print("Starting real-time inference & playback...")
# #     t0 = time.time()
# #     chunks = model.inference_stream(
# #         "Het heeft me veel tijd gekost om een stem te ontwikkelen, en nu ik hem heb, ga ik niet stil blijven.",
# #         "nl",
# #         gpt_cond_latent,
# #         speaker_embedding
# #     )

# #     for i, chunk in enumerate(chunks):
# #         if i == 0:
# #             print(f" Time to first chunk: {time.time() - t0:.2f} sec")
# #         print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

# #         # Save chunk to temporary WAV
# #         temp_file = "temp_chunk.wav"
# #         torchaudio.save(temp_file, chunk.squeeze().unsqueeze(0).cpu(), 24000)

# #         # Play chunk using aplay
# #         subprocess.run(["aplay", "-q", temp_file])

# #     print(" Finished rplayback.")

# # if __name__ == "__main__":
# #     main()
# import os
# import time
# import torch
# import subprocess
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import numpy as np

# def main():
#     print("Loading model...")

#     # Load config
#     config = XttsConfig()
#     config.load_json("/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained/config.json")

#     # Init model
#     model = Xtts.init_from_config(config)
#     model.load_checkpoint(
#         config,
#         checkpoint_dir="xtts_pretrained",
#         use_deepspeed=False
#     )
#     model.cuda()

#     print("Computing speaker latents...")
#     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
#         audio_path=["/home/rajan/Prabal_procit_works/XTTS streaming/00020.wav"]
#     )

#     print("Starting real-time inference & playback...")
#     t0 = time.time()
#     chunks = model.inference_stream(
#         "Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie. Deze implementatie toont precies hoe lang elke tekstgeneratie duurt. U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken. De timing analyse geeft inzicht in de prestaties van elk segment. Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt. Elke chunk wordt individueel verwerkt met gedetailleerde tijdsregistratie. Dit helpt bij het optimaliseren van de streaming prestaties in real-time. Streaming zorgt voor lagere perceptuele vertraging voor de gebruiker.",
#         "nl",
#         gpt_cond_latent,
#         speaker_embedding
#     )

#     # Launch ffplay subprocess for raw audio
#     ffplay_cmd = [
#         "ffplay",
#         "-f", "f32le",        # 32-bit float PCM
#         "-ar", "24000",       # sample rate
#         "-ac", "1",           # mono
#         "-nodisp",
#         "-autoexit",
#         "-loglevel", "quiet",
#         "pipe:0"              # read from stdin
#     ]
#     proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

#     for i, chunk in enumerate(chunks):
#         if i == 0:
#             print(f"Time to first chunk: {time.time() - t0:.2f} sec")
#         print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

#         # Convert to float32 numpy array
#         audio_np = chunk.squeeze().cpu().numpy().astype(np.float32)

#         # Write raw audio to ffplay stdin
#         proc.stdin.write(audio_np.tobytes())

#     # Close stdin to signal ffplay to finish
#     proc.stdin.close()
#     proc.wait()

#     print("Finished playback.")

# if __name__ == "__main__":
#     main()

