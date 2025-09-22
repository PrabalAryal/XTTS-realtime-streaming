# import os
# import time
# import torch
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import io
# import numpy as np

# app = FastAPI()
# #changed 
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# model = None
# gpt_cond_latent = None  
# speaker_embedding = None

# @app.on_event("startup")
# async def load_model():
#     global model, gpt_cond_latent, speaker_embedding
#     print("Loading model...")
#     config = XttsConfig()
#     config.load_json("/home/ai001/STC-AI/xtts_streaming/xtts_pretrained/config.json")
#     model = Xtts.init_from_config(config)
#     model.load_checkpoint(config, checkpoint_dir="/home/ai001/STC-AI/xtts_streaming/xtts_pretrained", use_deepspeed=False)
#     model.cuda()
#     print("Computing speaker latents...")
#     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
#         audio_path=["/home/ai001/STC-AI/xtts_streaming/00020.wav"]
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
            
#             audio_np = chunk.squeeze().cpu().numpy().astype(np.float32)
#             yield audio_np.tobytes()
#     return StreamingResponse(generate(), media_type="application/octet-stream", headers={"X-Accel-Buffering": "no"})#changed
#     #return StreamingResponse(generate(), media_type="application/octet-stream")
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)

# filepath: /home/rajan/Prabal_procit_works/XTTS streaming/server.py
# import os
# import time
# import torch
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import io
# import numpy as np
# from scipy.signal import resample

# app = FastAPI()
# #changed 
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# model = None
# gpt_cond_latent = None  
# speaker_embedding = None

# @app.on_event("startup")
# async def load_model():
#     global model, gpt_cond_latent, speaker_embedding
#     print("Loading model...")
#     config = XttsConfig()
#     config.load_json("/home/ai001/STC-AI/xtts_streaming/xtts_pretrained/config.json")
#     model = Xtts.init_from_config(config)
#     model.load_checkpoint(config, checkpoint_dir="/home/ai001/STC-AI/xtts_streaming/xtts_pretrained", use_deepspeed=False)
#     model.cuda()
#     print("Computing speaker latents...")
#     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
#         audio_path=["/home/ai001/STC-AI/xtts_streaming/00020.wav"]
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
            
#             audio_np = chunk.squeeze().cpu().numpy().astype(np.float32)
#             downsampled_audio = resample(audio_np, int(len(audio_np) * 8000 / 24000))
#             yield downsampled_audio.astype(np.float32).tobytes()
#     return StreamingResponse(generate(), media_type="application/octet-stream", headers={"X-Accel-Buffering": "no"})#changed
#     #return StreamingResponse(generate(), media_type="application/octet-stream")
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000) 
# import os
# import time
# import torch
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import io,re
# import numpy as np
# from scipy.signal import resample

# app = FastAPI()
# #changed 
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


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

# def split_text(text, max_len=200):
#     """
#     Split long text into sub-sentences safely.
#     First by punctuation, then by length.
#     """
#     sentences = re.split(r'(?<=[.!?,;:]) +', text)
#     chunks, current = [], ""
#     for s in sentences:
#         if len(current) + len(s) < max_len:
#             current += " " + s
#         else:
#             if current:
#                 chunks.append(current.strip())
#             current = s
#     if current:
#         chunks.append(current.strip())
#     return chunks


# @app.post("/inference_stream")
# async def stream_inference(text: str, language: str):
#     def generate():
#         sr = 24000
#         frame_size = int(0.02 * sr)  # 20 ms frames
#         tiny_pause = np.zeros(int(0.12 * sr), dtype=np.float32)  # 120 ms pause

#         torch.manual_seed(42)

#         # Step 1: Split text into short sub-chunks
#         text_chunks = split_text(text, max_len=200)
#         print(f"Split into {len(text_chunks)} sub-chunks.")

#         for idx, part in enumerate(text_chunks):
#             print(f"[Chunk {idx+1}/{len(text_chunks)}] {part[:60]}...")

#             # Step 2: Generate audio for the sub-chunk
#             chunks = model.inference_stream(part, language, gpt_cond_latent, speaker_embedding)

#             for chunk in chunks:
#                 audio_np = chunk.squeeze().cpu().numpy().astype(np.float32)

#                 # Step 3: Stream audio in 20ms frames
#                 for start in range(0, len(audio_np), frame_size):
#                     frame = audio_np[start:start + frame_size]
#                     if len(frame) == 0:
#                         continue
#                     yield frame.tobytes()

#             # Step 4: Tiny unnoticeable pause between chunks
#             if idx < len(text_chunks) - 1:
#                 yield tiny_pause.tobytes()

#     return StreamingResponse(
#         generate(),
#         media_type="application/octet-stream",
#         headers={"X-Accel-Buffering": "no"}
#     )

#     #return StreamingResponse(generate(), media_type="application/octet-stream")
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)

# filepath: /home/rajan/Prabal_procit_works/XTTS streaming/server.py
import os
import time
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import io
import numpy as np

app = FastAPI()
#changed 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
gpt_cond_latent = None  
speaker_embedding = None

@app.on_event("startup")
async def load_model():
    global model, gpt_cond_latent, speaker_embedding
    print("Loading model...")
    config = XttsConfig()
    config.load_json("/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/home/rajan/Prabal_procit_works/XTTS streaming/xtts_pretrained", use_deepspeed=False)
    model.cuda()
    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=["/home/rajan/Prabal_procit_works/XTTS streaming/00020.wav"]
    )
    print("Model and latents ready.")

@app.get("/compute_latents")
async def get_latents():
    return {"gpt_cond_latent": gpt_cond_latent.cpu().numpy().tolist(), "speaker_embedding": speaker_embedding.cpu().numpy().tolist()}

SAMPLE_RATE = 24000  # XTTS model's sample rate

@app.post("/inference_stream")
async def stream_inference(text: str, language: str):
    def generate():
        chunks = model.inference_stream(text, language, gpt_cond_latent, speaker_embedding)
        
        # We need a reference point for time
        last_chunk_time = time.time()
        
        for i, chunk in enumerate(chunks):
            # Convert the chunk to numpy and float32
            audio_np = chunk.squeeze().cpu().numpy().astype(np.float32)

            # Calculate the expected duration of this audio chunk in seconds
            # audio_np.shape[-1] gives the number of samples in the chunk
            chunk_duration = audio_np.shape[-1] / SAMPLE_RATE

            # Calculate how much time has passed since the last chunk was sent
            current_time = time.time()
            time_since_last_chunk = current_time - last_chunk_time
            
            # Calculate the time we need to wait to pace the stream
            # This is the expected duration minus the time it took to generate
            # a non-negative value.
            sleep_time = max(0, chunk_duration - time_since_last_chunk)

            # Wait to pace the stream
            if sleep_time > 0:
                print(f"Pacing: Sleeping for {sleep_time:.4f} seconds to match audio duration.")
                time.sleep(sleep_time)

            # Update the last chunk time for the next iteration
            last_chunk_time = time.time()
            
            # Yield the chunk's bytes
            yield audio_np.tobytes()

    return StreamingResponse(generate(), media_type="application/octet-stream", headers={"X-Accel-Buffering": "no"})

    #return StreamingResponse(generate(), media_type="application/octet-stream")
if __name__ == "__main__":
     import uvicorn
     uvicorn.run(app, port=8000)