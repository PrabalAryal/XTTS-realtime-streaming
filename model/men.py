# filepath: /home/rajan/Prabal_procit_works/XTTS streaming/server.py
# import os
# import time
# import torch
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import io
# import numpy as np

# app = FastAPI()


# model = None
# gpt_cond_latent = None  
# speaker_embedding = None

# @app.on_event("startup")
# async def load_model():
#     global model, gpt_cond_latent, speaker_embedding
#     print("Loading model...")
#     config = XttsConfig()
#     config.load_json("/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained/config.json")
#     model = Xtts.init_from_config(config)
#     model.load_checkpoint(config, checkpoint_dir="/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained", use_deepspeed=False)
#     model.cuda()
#     print("Computing speaker latents...")
#     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
#         audio_path=["/home/rajan/Prabal_procit_works/XTTS streaming/00020.wav"]
#     )
#     print("Model and latents ready.")

# @app.get("/compute_latents")
# async def get_latents():
#     return {"gpt_cond_latent": gpt_cond_latent.cpu().numpy().tolist(), "speaker_embedding": speaker_embedding.cpu().numpy().tolist()}

# @app.post("/inference_stream")
# async def stream_inference(text: str, language: str):
#     def generate():
#         t0 = time.time()
#         chunks = model.inference_stream(text, language, gpt_cond_latent, speaker_embedding)
#         for i, chunk in enumerate(chunks):
#             if i == 0:
#                 print(f"Time to first chunk: {time.time() - t0:.2f} sec")
#             print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
#             # Serialize chunk as bytes
#             audio_np = chunk.squeeze().cpu().numpy().astype(np.float32)
#             yield audio_np.tobytes()
#             #add delay to simulate real-time streaming
#             time.sleep(0.02)
#     return StreamingResponse(generate(), media_type="application/octet-stream")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

# import os
# import time
# import torch
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import numpy as np
# import wave
# import io
# import re

# app = FastAPI()

# model = None
# gpt_cond_latent = None  
# speaker_embedding = None

# # Maximum characters per chunk (can adjust based on GPT token limit)
# MAX_CHARS_PER_CHUNK = 40  

# @app.on_event("startup")
# async def load_model():
#     global model, gpt_cond_latent, speaker_embedding
#     print("Loading model...")
#     config = XttsConfig()
#     config.load_json("/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained/config.json")
#     model = Xtts.init_from_config(config)
#     model.load_checkpoint(config, checkpoint_dir="/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained", use_deepspeed=False)
#     model.cuda()
#     print("Computing speaker latents...")
#     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
#         audio_path=["/home/rajan/Prabal_procit_works/XTTS streaming/combined.wav"]
#     )
#     print("Model and latents ready.")

# @app.get("/compute_latents")
# async def get_latents():
#     return {"gpt_cond_latent": gpt_cond_latent.cpu().numpy().tolist(), "speaker_embedding": speaker_embedding.cpu().numpy().tolist()}

# @app.post("/inference_stream")
# async def stream_inference(text: str, language: str):
#     max_chars_per_request = 88  # approximate chunk size
#     punctuation_regex = re.compile(r'[.,;!? ]')#add space in the regex if needed to do this write r'[.,;!? ]'

#     # Function to split text by punctuation after max_chars_per_request
#     def split_text_by_punctuation(text, max_len):
#         chunks = []
#         start = 0
#         while start < len(text):
#             if len(text) - start <= max_len:
#                 chunks.append(text[start:])
#                 break
#             # Look for punctuation after max_len
#             match = punctuation_regex.search(text, start + max_len)
#             if match:
#                 end = match.end()  # include the punctuation
#             else:
#                 end = start + max_len  # fallback: just cut
#                  #To avoid breaking words, find the last space before end
#                 last_space = text.rfind(' ', start, end)
#                 if last_space > start:
#                     end = last_space + 1  # include the space
#             chunks.append(text[start:end].strip())
#             start = end
#         return chunks

#     text_chunks = split_text_by_punctuation(text, max_chars_per_request)

#     def generate():
#         for idx, chunk_text in enumerate(text_chunks):
#             print(f"Processing chunk {idx+1}/{len(text_chunks)}: {chunk_text[:30]}...")
#             t0 = time.time()
#             audio_chunks = model.inference_stream(chunk_text, language, gpt_cond_latent, speaker_embedding)
#             for i, audio_chunk in enumerate(audio_chunks):
#                 if i == 0:
#                     print(f"Time to first audio chunk of this text chunk: {time.time() - t0:.2f} sec")
#                 audio_np = audio_chunk.squeeze().cpu().numpy().astype(np.float32)
                
#                 yield audio_np.tobytes()
               
#             print(f"Chunk {idx+1}/{len(text_chunks)} processed in {time.time() - t0:.2f} sec")

#     return StreamingResponse(generate(), media_type="application/octet-stream")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
#........................above is the original code.........................


# import os
# import time
# import torch
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import numpy as np
# import struct
# import re

# app = FastAPI()

# model = None
# gpt_cond_latent = None  
# speaker_embedding = None

# # Maximum characters per chunk
# MAX_CHARS_PER_CHUNK = 88  

# @app.on_event("startup")
# async def load_model():
#     global model, gpt_cond_latent, speaker_embedding
#     print("Loading model...")
#     config = XttsConfig()
#     config.load_json("/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained/config.json")
#     model = Xtts.init_from_config(config)
#     model.load_checkpoint(
#         config,
#         checkpoint_dir="/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained",
#         use_deepspeed=False
#     )
#     model.cuda()
#     print("Computing speaker latents...")
#     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
#         audio_path=["/home/rajan/Prabal_procit_works/XTTS streaming/combined.wav"]
#     )
#     print("Model and latents ready.")

# @app.get("/compute_latents")
# async def get_latents():
#     return {
#         "gpt_cond_latent": gpt_cond_latent.cpu().numpy().tolist(),
#         "speaker_embedding": speaker_embedding.cpu().numpy().tolist()
#     }

# def split_text_by_punctuation(text, max_len):
#     punctuation_regex = re.compile(r'[.,;!? ]')  
#     chunks = []
#     start = 0
#     while start < len(text):
#         if len(text) - start <= max_len:
#             chunks.append(text[start:])
#             break
#         match = punctuation_regex.search(text, start + max_len)
#         if match:
#             end = match.end()
#         else:
#             end = start + max_len
#             last_space = text.rfind(' ', start, end)
#             if last_space > start:
#                 end = last_space + 1
#         chunks.append(text[start:end].strip())
#         start = end
#     return chunks

# def wav_header(sample_rate=22050, bits_per_sample=16, channels=1):
#     """Create a WAV header with unknown data length (good for streaming)."""
#     datasize = 0xFFFFFFFF  # "infinite" size for streaming
#     o = bytes("RIFF", 'ascii')
#     o += struct.pack('<I', datasize)
#     o += bytes("WAVE", 'ascii')
#     o += bytes("fmt ", 'ascii')
#     o += struct.pack('<I', 16)  # Subchunk1 size
#     o += struct.pack('<H', 1)   # PCM format
#     o += struct.pack('<H', channels)
#     o += struct.pack('<I', sample_rate)
#     o += struct.pack('<I', sample_rate * channels * bits_per_sample // 8)
#     o += struct.pack('<H', channels * bits_per_sample // 8)
#     o += struct.pack('<H', bits_per_sample)
#     o += bytes("data", 'ascii')
#     o += struct.pack('<I', datasize)
#     return o

# @app.post("/inference_stream")
# async def stream_inference(text: str, language: str):
#     text_chunks = split_text_by_punctuation(text, MAX_CHARS_PER_CHUNK)

#     def generate():
#         # Send WAV header first
#         yield wav_header(sample_rate=22050, bits_per_sample=16, channels=1)

#         for idx, chunk_text in enumerate(text_chunks):
#             print(f"Processing chunk {idx+1}/{len(text_chunks)}: {chunk_text[:30]}...")
#             t0 = time.time()
#             audio_chunks = model.inference_stream(
#                 chunk_text, language, gpt_cond_latent, speaker_embedding
#             )
#             for i, audio_chunk in enumerate(audio_chunks):
#                 if i == 0:
#                     print(f"Time to first audio chunk of this text chunk: {time.time() - t0:.2f} sec")

#                 audio_np = audio_chunk.squeeze().cpu().numpy().astype(np.float32)
#                 # Convert float32 -> int16 PCM
#                 int16_audio = np.int16(np.clip(audio_np, -1.0, 1.0) * 32767)

#                 # Stream PCM (no per-chunk headers)
#                 yield int16_audio.tobytes()

#             print(f"Chunk {idx+1}/{len(text_chunks)} processed in {time.time() - t0:.2f} sec")

#     # This is now a valid continuous WAV stream
#     return StreamingResponse(generate(), media_type="audio/wav")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

#.............. the above code send s chhunk in the form of wav file.................


import os
import time
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np
import re
import scipy.signal

app = FastAPI()

model = None
gpt_cond_latent = None  
speaker_embedding = None

# Maximum characters per chunk
MAX_CHARS_PER_CHUNK = 88  

# Target sample rate for Asterisk
TARGET_SR = 8000

@app.on_event("startup")
async def load_model():
    global model, gpt_cond_latent, speaker_embedding
    print("Loading model...")
    config = XttsConfig()
    config.load_json("/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir="/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained",
        use_deepspeed=False
    )
    model.cuda()
    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=["/home/rajan/Prabal_procit_works/XTTS streaming/combined.wav"]
    )
    print("Model and latents ready.")

@app.get("/compute_latents")
async def get_latents():
    return {
        "gpt_cond_latent": gpt_cond_latent.cpu().numpy().tolist(),
        "speaker_embedding": speaker_embedding.cpu().numpy().tolist()
    }

def split_text_by_punctuation(text, max_len):
    punctuation_regex = re.compile(r'[.,;!? ]')  
    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_len:
            chunks.append(text[start:])
            break
        match = punctuation_regex.search(text, start + max_len)
        if match:
            end = match.end()
        else:
            end = start + max_len
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def resample_to_8k(audio_np, in_sr=24000, out_sr=8000):
    """Resample float32 audio from in_sr to out_sr and convert to int16 PCM"""
    audio_resampled = scipy.signal.resample_poly(audio_np, out_sr, in_sr)
    int16_audio = np.int16(np.clip(audio_resampled, -1.0, 1.0) * 32767)
    return int16_audio.tobytes()

@app.post("/inference_stream")
async def stream_inference(text: str, language: str):
    text_chunks = split_text_by_punctuation(text, MAX_CHARS_PER_CHUNK)

    def generate():
        total_bytes_sent = 0
        for idx, chunk_text in enumerate(text_chunks):
            print(f"Processing chunk {idx+1}/{len(text_chunks)}: {chunk_text[:30]}...")
            t0 = time.time()
            audio_chunks = model.inference_stream(
                chunk_text, language, gpt_cond_latent, speaker_embedding
            )
            for i, audio_chunk in enumerate(audio_chunks):
                if i == 0:
                    print(f"Time to first audio chunk: {time.time() - t0:.2f} sec")

                audio_np = audio_chunk.squeeze().cpu().numpy().astype(np.float32)
                pcm_bytes = resample_to_8k(audio_np, in_sr=24000, out_sr=TARGET_SR)
                num_samples = len(pcm_bytes) // 2  # 16-bit = 2 bytes per sample
                duration_ms = num_samples / TARGET_SR * 1000
                bit_depth = 16
                chunk_size_bytes = len(pcm_bytes)
                min_amp = np.min(np.frombuffer(pcm_bytes, dtype=np.int16))
                max_amp = np.max(np.frombuffer(pcm_bytes, dtype=np.int16))
                total_bytes_sent += chunk_size_bytes

                print(f"  -> Chunk {i+1} stats: samples={num_samples}, "
                      f"duration={duration_ms:.1f} ms, bits={bit_depth}, "
                      f"packet_size={chunk_size_bytes} bytes, min={min_amp}, max={max_amp}, "
                      f"total_sent={total_bytes_sent} bytes")
                # Stream raw 16-bit PCM at 8kHz
                yield pcm_bytes

            print(f"Chunk {idx+1}/{len(text_chunks)} processed in {time.time() - t0:.2f} sec")

    # Use application/octet-stream since this is raw PCM

    return StreamingResponse(generate(), media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
