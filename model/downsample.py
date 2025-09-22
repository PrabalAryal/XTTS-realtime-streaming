import time
import subprocess
import requests
import numpy as np
import os
from scipy.io import wavfile

def main():
    # API base URL
    base_url = "http://172.20.69.222:8007"

    # Optional: Fetch latents (though server handles it internally)
    # response = requests.get(f"{base_url}/compute_latents")
    # latents = response.json()  # Not used in this example, as server manages it

    # List of sentences to process in a loop
    sentences = [
        "Hoewel het buiten donker en regenachtig was, besloot ik toch een lange wandeling te maken door het park.",
        "Omdat ik vond dat de frisse lucht en het geluid van de vallende regendruppels op de bladeren een rustgevend effect hadden.",
        "En bovendien gaf het mij de kans om mijn gedachten te ordenen en even te ontsnappen aan de drukte van de dag.",
        "Ik ben blij dat ik die beslissing heb genomen."
    ]
    language = "nl"

    # Create directory for saving audio
    output_dir = "baud1_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Starting real-time inference & playback...")
    t0 = time.time()

    for sentence in sentences:
        print(f"Processing sentence: {sentence[:50]}...")

        # Accumulate downsampled audio data for this sentence
        sentence_audio_data = b""

        # Launch ffplay subprocess for each sentence to avoid pipe issues
        ffplay_cmd = [
            "ffplay",
            "-f", "f32le",        # 32-bit float PCM
            "-ar", "48000",        # Downsampled sample rate
            "-ac", "1",           # mono
            "-nodisp",
            "-autoexit",
            "-loglevel", "quiet",
            "-fflags", "nobuffer",  # Disable buffering for lower latency
            "-bufsize", "0",       # Minimal buffer size
            "-probesize", "32",    # Small probe size
            "pipe:0"              # read from stdin
        ]
        proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

        # Stream inference from API for each sentence
        response = requests.post(f"{base_url}/inference_stream", params={"text": sentence, "language": language}, stream=True)
        if response.status_code != 200:
            print("Error in API request")
            return

        chunk_size = 4 *8000  # Further reduced chunk size for lower latency
        for chunk_bytes in response.iter_content(chunk_size=chunk_size):
            if chunk_bytes:
                # Use sox for faster downsampling from 24kHz to 8kHz
                sox_proc = subprocess.Popen(
                    ['sox', '-t', 'raw', '-r', '24000', '-c', '1', '-e', 'float', '-b', '32', '-', '-t', 'raw', '-r', '48000', '-c', '1', '-e', 'float', '-b', '32', '-'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL  # Suppress sox output
                )
                sox_proc.stdin.write(chunk_bytes)
                sox_proc.stdin.close()
                downsampled_bytes = sox_proc.stdout.read()
                sox_proc.wait()

                # Write downsampled bytes to ffplay stdin
                if proc.poll() is None:
                    try:
                        proc.stdin.write(downsampled_bytes)
                        proc.stdin.flush()  # Flush to ensure data is sent
                    except BrokenPipeError:
                        print("Broken pipe: ffplay may have exited. Stopping playback for this sentence.")
                        break
                else:
                    print("ffplay process has exited. Stopping playback for this sentence.")
                    break

                # Accumulate for saving
                sentence_audio_data += downsampled_bytes

        # Close stdin to signal ffplay to finish for this sentence
        if proc.poll() is None:
            proc.stdin.close()
            proc.wait()

        # Save sentence audio after processing
        if sentence_audio_data:
            # Convert bytes to numpy float32 array
            audio_np = np.frombuffer(sentence_audio_data, dtype=np.float32)
            # Sanitize filename
            filename = sentence.replace(' ', '_').replace(',', '').replace('.', '').replace('?', '').replace('!', '') + '.wav'
            filepath = os.path.join(output_dir, filename)
            # Write as WAV with downsampled samples
            wavfile.write(filepath, 48000, audio_np)
            print(f"Audio saved as {filepath}.")

    print("Finished playback.")

if __name__ == "__main__":
    main()