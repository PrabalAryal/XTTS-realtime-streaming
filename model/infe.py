# # filepath: /home/rajan/Prabal_procit_works/XTTS streaming/client.py
# import time
# import subprocess
# import requests
# import numpy as np
# from io import BytesIO

# def main():
#     # API base URL
#     base_url = "http://127.0.0.1:8000"

#     # Optional: Fetch latents (though server handles it internally)
#     # response = requests.get(f"{base_url}/compute_latents")
#     # latents = response.json()  # Not used in this example, as server manages it

#     # text = "Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie.Deze implementatie toont precies hoe lang elke tekstgeneratie duurt.U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken.De timing analyse geeft inzicht in de prestaties van elk segment.Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt.Elke chunk wordt individueel verwerkt met gedetailleerde tijdsregistratie. Dit helpt bij het optimaliseren van de streaming prestaties in real-time.Streaming zorgt voor lagere perceptuele vertraging voor de gebruiker."
#     text="Hoewel het buiten donker en regenachtig was, besloot ik toch een lange wandeling te maken door het park, omdat ik vond dat de frisse lucht en het geluid van de vallende regendruppels op de bladeren een rustgevend effect hadden, en bovendien gaf het mij de kans om mijn gedachten te ordenen en even te ontsnappen aan de drukte van de dag. Ik ben blij dat ik die beslissing heb genomen."
#     language = "nl"

#     print("Starting real-time inference & playback...")
#     t0 = time.time()

#     # Stream inference from API
#     response = requests.post(f"{base_url}/inference_stream", params={"text": text, "language": language}, stream=True)
#     if response.status_code != 200:
#         print("Error in API request")
#         return

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

#     chunk_size = 4 * 24000  # Approximate bytes per chunk (adjust based on model output)
#     for chunk_bytes in response.iter_content(chunk_size=chunk_size):
#         if chunk_bytes:
#             # Write raw audio to ffplay stdin
#             proc.stdin.write(chunk_bytes)

#     # Close stdin to signal ffplay to finish
#     proc.stdin.close()
#     proc.wait()
#     print("Finished playback.")

# if __name__ == "__main__":
#     main()

# import time
# import subprocess
# import requests
# import numpy as np
# import os
# from scipy.io import wavfile
# import uuid  # For unique ID

# def main():
#     base_url = "http://127.0.0.1:8000"

#     sentences = [
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie. Deze implementatie toont precies hoe lang elke tekstgeneratie duurt. U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken. De timing analyse geeft inzicht in de prestaties van elk segment. Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt.",
#     ]
#     language = "nl"

#     output_dir = "new_again_outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     print("Starting real-time inference & playback...")
#     t0 = time.time()

#     # ffplay command with small buffer for smooth long-sentence playback
#     ffplay_cmd = [
#         "ffplay",
#         "-f", "f32le",        # 32-bit float PCM
#         "-ar", "24000",       # sample rate
#         "-ac", "1",           # mono
#         "-nodisp",
#         "-autoexit",
#         "-loglevel", "quiet",
#         "-bufsize", "1024k",   # small buffer prevents speed-up
#         "pipe:0"
#     ]
#     proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

#     frame_size = 480  # 20ms frames at 24kHz

#     for i, sentence in enumerate(sentences):
#         print(f"Processing sentence {i+1}: {sentence[:50]}...")
#         sentence_start = time.time()
#         sentence_audio_data = b""

#         response = requests.post(
#             f"{base_url}/inference_stream",
#             params={"text": sentence, "language": language},
#             stream=True
#         )
#         if response.status_code != 200:
#             print("Error in API request")
#             return

#         # Stream audio in steady 20ms frames
#         for chunk_bytes in response.iter_content(chunk_size=frame_size*4):  # 4 bytes per float32
#             if chunk_bytes:
#                 audio_np = np.frombuffer(chunk_bytes, dtype=np.float32)

#                 for start in range(0, len(audio_np), frame_size):
#                     frame = audio_np[start:start + frame_size]
#                     if len(frame) == 0:
#                         continue
#                     proc.stdin.write(frame.tobytes())
#                     proc.stdin.flush()

#                 # Accumulate for saving
#                 sentence_audio_data += chunk_bytes

#         sentence_time = time.time() - sentence_start
#         print(f"Sentence {i+1} processed in {sentence_time:.2f} sec.")

#         # Save audio after streaming
#         if sentence_audio_data:
#             audio_np = np.frombuffer(sentence_audio_data, dtype=np.float32)
#             timestamp = str(int(time.time()))
#             unique_id = str(uuid.uuid4())[:8]
#             filename = f"{timestamp}_{unique_id}.wav"
#             filepath = os.path.join(output_dir, filename)
#             wavfile.write(filepath, 24000, audio_np)
#             print(f"Audio saved as {filepath}.")

#     proc.stdin.close()
#     proc.wait()
#     print("Finished playback.")

# if __name__ == "__main__":
#     main()


# import time
# import subprocess
# import requests
# import numpy as np
# import os
# import uuid
# from scipy.io import wavfile

# def split_text(text, max_chars=250):
#     """Split text into chunks of max_chars, ideally at sentence or comma boundaries."""
#     import re
#     parts = re.split(r'([.,;!?])', text)
#     chunks = []
#     current_chunk = ""
#     for part in parts:
#         if len(current_chunk) + len(part) <= max_chars:
#             current_chunk += part
#         else:
#             if current_chunk:
#                 chunks.append(current_chunk.strip())
#             current_chunk = part
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks

# def stream_tts_chunk(proc, base_url, chunk, language, frame_size=480):
#     """Stream a single text chunk to ffplay in real-time."""
#     response = requests.post(
#         f"{base_url}/inference_stream",
#         params={"text": chunk, "language": language},
#         stream=True
#     )
#     if response.status_code != 200:
#         print("Error in API request")
#         return b""

#     audio_data = b""
#     for chunk_bytes in response.iter_content(chunk_size=frame_size*4):
#         if chunk_bytes:
#             audio_np = np.frombuffer(chunk_bytes, dtype=np.float32)
#             # Write each 20ms frame immediately
#             for start in range(0, len(audio_np), frame_size):
#                 frame = audio_np[start:start + frame_size]
#                 if len(frame) == 0:
#                     continue
#                 proc.stdin.write(frame.tobytes())
#                 proc.stdin.flush()
#             audio_data += chunk_bytes

#     # Optional: small silence between chunks
#     silence = np.zeros(int(0.05 * 24000), dtype=np.float32)
#     proc.stdin.write(silence.tobytes())
#     proc.stdin.flush()
#     audio_data += silence.tobytes()

#     return audio_data

# def main():
#     base_url = "http://127.0.0.1:8000"
#     sentences = [
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie. Deze implementatie toont precies hoe lang elke tekstgeneratie duurt. U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken. De timing analyse geeft inzicht in de prestaties van elk segment. Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt.",
#     ]
#     language = "nl"

#     output_dir = "new_again_outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     print("Starting real-time streaming TTS playback...")

#     # Start ffplay for live playback
#     ffplay_cmd = [
#         "ffplay",
#         "-f", "f32le",
#         "-ar", "24000",
#         "-ac", "1",
#         "-nodisp",
#         "-autoexit",
#         "-loglevel", "quiet",
#         "-bufsize", "1024k",
#         "pipe:0"
#     ]
#     proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

#     frame_size = 480  # 20ms frames at 24kHz

#     for i, sentence in enumerate(sentences):
#         print(f"\nProcessing sentence {i+1}: {sentence[:50]}...")
#         sentence_start = time.time()
#         sentence_audio_data = b""

#         chunks = split_text(sentence, max_chars=250)

#         for chunk_idx, chunk in enumerate(chunks):
#             print(f"  Streaming chunk {chunk_idx+1}/{len(chunks)}...")
#             audio_bytes = stream_tts_chunk(proc, base_url, chunk, language, frame_size)
#             sentence_audio_data += audio_bytes

#         sentence_time = time.time() - sentence_start
#         print(f"Sentence {i+1} fully streamed in {sentence_time:.2f} sec.")

#         # Save full sentence audio
#         if sentence_audio_data:
#             audio_np = np.frombuffer(sentence_audio_data, dtype=np.float32)
#             timestamp = str(int(time.time()))
#             unique_id = str(uuid.uuid4())[:8]
#             filename = f"{timestamp}_{unique_id}.wav"
#             filepath = os.path.join(output_dir, filename)
#             wavfile.write(filepath, 24000, audio_np)
#             print(f"Audio saved as {filepath}.")

#     proc.stdin.close()
#     proc.wait()
#     print("Finished playback.")

# if __name__ == "__main__":
#     main()

# import time
# import subprocess
# import requests
# import numpy as np
# import os
# from scipy.io import wavfile
# import uuid  # For unique ID

# def main():
#     # API base URL
#     base_url = "http://172.20.69.222:8007"

#     # List of sentences to process in a loop
#     sentences = [

#         "Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie. Deze implementatie toont precies hoe lang elke tekstgeneratie duurt. U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken. De timing analyse geeft inzicht in de prestaties van elk segment. Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt",

#     ]
#     language = "nl"

#     # Create directory for saving audio
#     output_dir = "new_again_outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     print("Starting real-time inference & playback...")
#     t0 = time.time()

#     # Launch ffplay subprocess for raw audio with low-latency options
#     ffplay_cmd = [
#         "ffplay",
#         "-f", "f32le",        # 32-bit float PCM
#         "-ar", "24000",       # Sample rate
#         "-ac", "1",           # mono
#         "-nodisp",
#         "-autoexit",
#         "-loglevel", "quiet",
#         "-fflags", "nobuffer",  # Disable buffering
#         "-bufsize", "50k",       # Minimal buffer size
#         "-probesize", "32",   # Small probe size

#         "pipe:0"              # read from stdin

#     ]
#     proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

#     for i, sentence in enumerate(sentences):
#         print(f"Processing sentence {i+1}: {sentence[:50]}...")
#         sentence_start = time.time()

#         # Accumulate audio data for this sentence
#         sentence_audio_data = b""

#         # Stream inference from API for each sentence
#         response = requests.post(f"{base_url}/inference_stream", params={"text": sentence, "language": language}, stream=True)
#         if response.status_code != 200:
#             print("Error in API request")
#             return

#         chunk_size = 4 * 480  # Reduced for more uniform chunks
#         chunk_count = 0
#         for chunk_bytes in response.iter_content(chunk_size=chunk_size):
#             if chunk_bytes:
#                 proc.stdin.write(chunk_bytes)
#                 proc.stdin.flush()  # Flush immediately
#                 sentence_audio_data += chunk_bytes
#                 chunk_count += 1

#         sentence_time = time.time() - sentence_start
#         # target_time = 3.5  # Target consistent time per sentence
#         # if sentence_time < target_time:
#         #     time.sleep(target_time - sentence_time)
#         print(f"Sentence {i+1} processed in {sentence_time:.2f} sec, {chunk_count} chunks.")

#         # Save sentence audio after processing
#         if sentence_audio_data:
#             # Convert to float32
#             audio_np = np.frombuffer(sentence_audio_data, dtype=np.float32)
            
#             # Generate unique filename with timestamp or UUID
#             timestamp = str(int(time.time()))  # Unix timestamp
#             unique_id = str(uuid.uuid4())[:8]  # Short UUID
#             filename = f"{timestamp}_{unique_id}.wav"  # e.g., 1699999999_12345678.wav
#             filepath = os.path.join(output_dir, filename)
            
#             wavfile.write(filepath, 24000, audio_np)
#             print(f"Audio saved as {filepath}.")

#     # Close stdin to signal ffplay to finish
#     proc.stdin.close()
#     proc.wait()
#     print("Finished playback.")

# if __name__ == "__main__":
#     main()

# ................the afforementioned code is for client.py to test the api of inference_stream in infe.py. this saves the audio file after playing too and uses ffpplay.................






# if __name__ == "__main__":
#     main()

# import requests
# import pyaudio
# import time


# BASE_URL = "http://127.0.0.1:8000/inference_stream"

# # BASE_URL = "http://172.20.69.222:8007/inference_stream"
# # URL = "http://127.0.0.1:8000/live-audio"

# text = "mijn naam is katherine. ik hou van programmeren."
# language = "nl"

# params = {"text": text, "language": language}

# # Match backend ffplay settings
# CHANNELS = 1
# RATE = 48000
# FORMAT = pyaudio.paFloat32   # <- 32-bit float PCM

# CHUNK = 1024

# p = pyaudio.PyAudio()
# stream = p.open(
#     format=FORMAT,
#     channels=CHANNELS,
#     rate=RATE,
#     output=True
# )

# start_time = time.time()
# chunk_count = 0

# with requests.post(BASE_URL, params=params, data="", stream=True) as r:
# # with requests.get(URL, data="", stream=True) as r:
#     print("Status:", r.status_code)

#     for chunk in r.iter_content():
#         if chunk:
#             stream.write(chunk)
#             chunk_count += 1
#             elapsed = time.time() - start_time
#             print(f"Chunk {chunk_count} received, Elapsed: {elapsed:.2f}s", end="\r")

# stream.stop_stream()
# stream.close()
# p.terminate()

# print(f"\nFinished. Total chunks: {chunk_count}, Total time: {time.time()-start_time:.2f}s")


# import requests
# import pyaudio
# import numpy as np
# import time

# # Server endpoint
# BASE_URL = "http://172.20.69.222:8007/inference_stream"

# # Text and language to synthesize
# text = "mijn naam is katherine. ik hou van programmeren."
# language = "nl"
# params = {"text": text, "language": language}

# # Audio parameters
# CHANNELS = 1
# RATE = 24000        # matches XTTS backend output
# FORMAT = pyaudio.paFloat32
# FRAME_SIZE = 1024   # samples per frame

# # Initialize PyAudio stream
# p = pyaudio.PyAudio()
# stream = p.open(
#     format=FORMAT,
#     channels=CHANNELS,
#     rate=RATE,
#     output=True
# )

# start_time = time.time()
# chunk_count = 0

# # Stream audio from backend
# with requests.post(BASE_URL, params=params, stream=True) as r:
#     print("Status:", r.status_code)

#     if r.status_code == 200:
#         buffer = b""
#         for chunk in r.iter_content(chunk_size=FRAME_SIZE * 4):  # 4 bytes per float32
#             if chunk:
#                 buffer += chunk
#                 # Play only complete frames
#                 while len(buffer) >= FRAME_SIZE * 4:
#                     frame_bytes = buffer[:FRAME_SIZE * 4]
#                     buffer = buffer[FRAME_SIZE * 4:]
#                     audio_np = np.frombuffer(frame_bytes, dtype=np.float32)
#                     stream.write(audio_np.tobytes())
#                     chunk_count += 1
#                     elapsed = time.time() - start_time
#                     print(f"Chunk {chunk_count} received, Elapsed: {elapsed:.2f}s", end="\r")

#         # Play any leftover samples (less than FRAME_SIZE)
#         if len(buffer) >= 4:
#             remainder_samples = np.frombuffer(buffer, dtype=np.float32)
#             stream.write(remainder_samples.tobytes())
#             chunk_count += 1

# stream.stop_stream()
# stream.close()
# p.terminate()

# print(f"\nFinished. Total chunks: {chunk_count}, Total time: {time.time()-start_time:.2f}s")


# import requests
# import subprocess
# import time

# # Backend URL
# BASE_URL = "http://172.20.69.222:8007/inference_stream"
# # BASE_URL="http://127.0.0.1:8000/inference_stream"

# # Text to synthesize
# # text='Herken'
# text = "Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie. Deze implementatie toont precies hoe lang elke tekstgeneratie duurt. U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken. De timing analyse geeft inzicht in de prestaties van elk segment. Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt.Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie. Deze implementatie toont precies hoe lang elke tekstgeneratie duurt. U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken. De timing analyse geeft inzicht in de prestaties van elk segment. Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt.Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie. Deze implementatie toont precies hoe lang elke tekstgeneratie duurt. U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken. De timing analyse geeft inzicht in de prestaties van elk segment. Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt.Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie. Deze implementatie toont precies hoe lang elke tekstgeneratie duurt. U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken. De timing analyse geeft inzicht in de prestaties van elk segment. Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt."
# language = "nl"
# params = {"text": text, "language": language}

# # Audio parameters (matches backend)
# CHANNELS = 1
# RATE = 24000
# FORMAT = 'float'  # Sox encoding
# BITS = 32         # 32-bit float

# # sox command for real-time playback
# sox_cmd = [
#     "play",
#     "-t", "raw",         # raw PCM input
#     "-r", str(RATE),     # sample rate
#     "-e", FORMAT,        # encoding
#     "-b", str(BITS),     # bits per sample
#     "-c", str(CHANNELS), # channels
#     "-",                  # read from stdin
#     "-q",                 # quiet
#     "trim", "0"           # start immediately
# ]

# # Launch sox subprocess
# proc = subprocess.Popen(sox_cmd, stdin=subprocess.PIPE)

# start_time = time.time()
# chunk_count = 0

# # Stream audio from backend
# with requests.post(BASE_URL, params=params, stream=True) as r:
#     print("Status:", r.status_code)

#     if r.status_code == 200:
#         for chunk in r.iter_content(chunk_size=1024 * 4):  # 4 KB chunks
#             if chunk:
#                 proc.stdin.write(chunk)
#                 proc.stdin.flush()  # immediate playback
#                 chunk_count += 1
#                 elapsed = time.time() - start_time
#                 print(f"Chunk {chunk_count} received, Elapsed: {elapsed:.2f}s", end="\r")

# # Close sox stdin to finish playback
# proc.stdin.close()
# proc.wait()

# print(f"\nFinished. Total chunks: {chunk_count}, Total time: {time.time() - start_time:.2f}s")

#....................the afforementioned commented code uses sox for playing the audio ....................

import requests
import subprocess
import time
import numpy as np

# Backend URL
BASE_URL = "http://127.0.0.1:8000/inference_stream"

text = "Goedemorgen. Ik ben Annemarie, de digitale medewerker. Om u goed en snel van dienst te zijn, vragen wij u om een aantal vragen te beantwoorden. Hierdoor kunnen mijn collega's u beter helpen. Wilt u mijn vragen beantwoorden?"  # truncated
language = "nl"
params = {"text": text, "language": language}

# ffplay command for real-time playback
ffplay_cmd = [
    "ffplay",
    "-f", "s16le",
    "-ar", "8000",
    "-ac", "1",
    "-nodisp",
    "-autoexit",
    "-loglevel", "quiet",
    "pipe:0"
]

proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
chunk_count = 0
start_time = time.time()

with requests.post(BASE_URL, params=params, stream=True) as r:
    print("Status:", r.status_code)

    if r.status_code == 200:
        for chunk in r.iter_content(chunk_size=1024 * 4):
            if chunk:
                proc.stdin.write(chunk)
                proc.stdin.flush()
                chunk_count += 1

                # Verify if chunk is s16le
                if len(chunk) % 2 != 0:
                    print(f"Warning: Chunk {chunk_count} length {len(chunk)} is not multiple of 2 bytes!")

                # Try interpreting as int16 little-endian
                try:
                    audio_samples = np.frombuffer(chunk, dtype='<i2')  # little-endian int16
                    min_val = audio_samples.min()
                    max_val = audio_samples.max()
                    print(f"Chunk {chunk_count}: size={len(chunk)} bytes, min={min_val}, max={max_val}, assumed s16le")
                except Exception as e:
                    print(f"Chunk {chunk_count}: cannot interpret as s16le! Error: {e}")

proc.stdin.close()
proc.wait()
print(f"\nFinished. Total chunks: {chunk_count}, Total time: {time.time() - start_time:.2f}s")
