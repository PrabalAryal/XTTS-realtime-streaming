# # # filepath: /home/rajan/Prabal_procit_works/XTTS streaming/client.py
# # import time
# # import subprocess
# # import requests
# # import numpy as np
# # from io import BytesIO

# # def main():
# #     # API base URL
# #     base_url = "http://127.0.0.1:8000"

# #     # Optional: Fetch latents (though server handles it internally)
# #     # response = requests.get(f"{base_url}/compute_latents")
# #     # latents = response.json()  # Not used in this example, as server manages it

# #     # text = "Hallo en welkom bij onze gedetailleerde streaming TTS demonstratie.Deze implementatie toont precies hoe lang elke tekstgeneratie duurt.U ziet dat kleinere stukken sneller worden verwerkt dan grotere stukken.De timing analyse geeft inzicht in de prestaties van elk segment.Merk op dat er geen voorafgaande generatie van alle audio plaatsvindt.Elke chunk wordt individueel verwerkt met gedetailleerde tijdsregistratie. Dit helpt bij het optimaliseren van de streaming prestaties in real-time.Streaming zorgt voor lagere perceptuele vertraging voor de gebruiker."
# #     text="Hoewel het buiten donker en regenachtig was, besloot ik toch een lange wandeling te maken door het park, omdat ik vond dat de frisse lucht en het geluid van de vallende regendruppels op de bladeren een rustgevend effect hadden, en bovendien gaf het mij de kans om mijn gedachten te ordenen en even te ontsnappen aan de drukte van de dag. Ik ben blij dat ik die beslissing heb genomen."
# #     language = "nl"

# #     print("Starting real-time inference & playback...")
# #     t0 = time.time()

# #     # Stream inference from API
# #     response = requests.post(f"{base_url}/inference_stream", params={"text": text, "language": language}, stream=True)
# #     if response.status_code != 200:
# #         print("Error in API request")
# #         return

# #     # Launch ffplay subprocess for raw audio
# #     ffplay_cmd = [
# #         "ffplay",
# #         "-f", "f32le",        # 32-bit float PCM
# #         "-ar", "24000",       # sample rate
# #         "-ac", "1",           # mono
# #         "-nodisp",
# #         "-autoexit",
# #         "-loglevel", "quiet",
# #         "pipe:0"              # read from stdin
# #     ]
# #     proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

# #     chunk_size = 4 * 24000  # Approximate bytes per chunk (adjust based on model output)
# #     for chunk_bytes in response.iter_content(chunk_size=chunk_size):
# #         if chunk_bytes:
# #             # Write raw audio to ffplay stdin
# #             proc.stdin.write(chunk_bytes)

# #     # Close stdin to signal ffplay to finish
# #     proc.stdin.close()
# #     proc.wait()
# #     print("Finished playback.")

# # if __name__ == "__main__":
# #     main()

import time
import subprocess
import requests
import numpy as np
import os
from scipy.io import wavfile

def main():
    #........................................................... API base URL............................
    base_url = "http://127.0.0.1:8000"

    # .................................................List of sentences to inference.....................
    sentences = [
"Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
"Het kan gas of een ondergelopen kelder maar natuurlijk zijn er ook andere mogelijkheden."
"Mechanisme-monteurs maken met minimale middelen maximale mogelijkheden mogelijk.",
"Moeilijke mechanisme-managers manipuleren met mechanische maatwerk-magie.",
"maar natuurlijk zijn er ook andere mogelijkheden",
"maar er zijn natuurlijk ook andere mogelijkheden",
"ik zie de champignons schuifelen richting de waswachine",
"Scherpe schaduwen schuiven schuifelend over het schoolplein. Charmante chefs kiezen champignons voor chique chardonnay-chutney.",
"testzin woorden waar ch wel als g wordt uitgesproken",
"Testzinn woorden met ch die als sj worden uitgesproken",
"Mag ik uw gegevens noteren?",
"Betreft het een probleem in de eigen woning of algemene ruimte?",
"Melding doorgeven aan de dienstdoende van Staedion",
"Ga in de eerste melding verder en noteer in deze melding het meldingsnummer van die melding",
"Melding op Geen fax email",
"In welke ruimte is het probleem?",
"Dit antwoord herken ik niet. Veel voorkomende keuzes zijn bijvoorbeeld de badkamer, balkon of berging maar natuurlijk zijn er ook andere mogelijkheden.Kunt u het antwoord nogmaals geven?",
"Dit antwoord herken ik niet. Veel voorkomende keuzes zijn bijvoorbeeld het dak,de galerij of de gang maar natuurlijk zijn er ook andere mogelijkheden. Kunt u het antwoord nogmaals geven?",
"Wat is het probleem in de badkamer?",
"Dit antwoord herken ik niet. Veel voorkomende keuzes zijn bijvoorbeeld de afvoer, elektra of het gas maar natuurlijk zijn er ook andere mogelijkheden. Kunt u het antwoord nogmaals geven?",
"Gaat het om een lekkage of een verstopping?",
"Dit antwoord herken ik niet. Kunt u het antwoord nogmaals geven? Maak een keuze uit lekkage of verstopping.",
"Wat is er verstopt?",
"Dit antwoord herken ik niet. Kunt u het antwoord nogmaals geven? Maak een keuze uit wastafel, douche, wasmachine of toilet.",
"Adviseer om de wastafel douche wasmachine niet te gebruiken men kan de volgende werkdag opnieuw bellen",
"Is er een tweede toilet in de woning?",
"Dit antwoord herken ik niet. Kunt u het antwoord nogmaals geven? Maak een keuze uit ja of nee.",
"Adviseer het verstopte toilet niet te gebruiken en verwijs de bewoner door naar de volgende werkdag",
"Bewoner kan contact opnemen met het ontstoppingsbedrijf.",
"Wat is er lek?",
"Dit antwoord herken ik niet. Veel voorkomende keuzes zijn bijvoorbeeld de kraan,toilet stortbak of afvoer maar natuurlijk zijn er ook andere mogelijkheden. Kunt u het antwoord nogmaals geven?",

"Adviseer de kraan niet te gebruiken en adviseer de bewoner de volgende werkdag opnieuw te bellen of een mail te sturen.",
"Adviseer de bewoner de stortbak af te sluiten doormiddel van het kraantje.",
"Melding doorgeven aan de loodgieter",
"Geen spoed adviseer de bewoner de wastafel of douche niet te gebruiken en de volgende werkdag te bellen",
"Is dit gebeurd door eigen toedoen?",
"Meldt dan de de kosten voor de bewoner zijn, melding doorgeven aan dienstdoende van Staedion",
"melding doorgeven aan dienstdoende van Staedion",
"Lukt het om de stortbak af te sluiten door gebruik van het kraantje?",
"Adviseer de bewoner om de volgende werkdag te bellen of een mail te sturen",
"Melding doorgeven aan de nooddienstmedewerker",
"Wat is er defect?",
"Dit antwoord herken ik niet. Veel voorkomende keuzes zijn bijvoorbeeld de toiletbril, kraan of douchekop maar natuurlijk zijn er ook andere mogelijkheden.Kunt u het antwoord nogmaals geven?",
"Geen spoed adviseer de bewoner om de volgende werkdag terug te bellen of een mail te sturen",
"Geen spoed adviseer de bewoner om de volgende werkdag terug te bellen",
"Wat is het probleem met de kraan?",
"Dit antwoord herken ik niet. Kunt u het antwoord nogmaals geven? Maak een keuze uit kraan druppelt of kraan kan niet dichtgedraaid.",
"Wat is het probleem op het balkon of dakterras?",
"Veel voorkomende keuzes zijn bijvoorbeeld het hang en sluitwerk van de deur,gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden.Kunt u het antwoord nogmaals geven?",
"Melding doorgeven aan het ontstoppingsbedrijf",
"Is er een gevaarlijke situatie ontstaan?",
"Melding doorgeven aan het gekoppelde bedrijf (bouwkundig)",
"Bewoner doorverwijzen naar de volgende werkdag",
"Wat is het probleem in de berging?",
"Veel voorkomende keuzes zijn bijvoorbeeld elektra, gas of een ondergelopen kelder maar natuurlijk zijn er ook andere mogelijkheden. Kunt u het antwoord nogmaals geven?",
"Melding doorgeven aan de firma Proper",
"Wat is het probleem in de buitenruimte of tuin?",
"Veel voorkomende keuzes zijn bijvoorbeeld losse tegels of een kapotte schutting maar natuurlijk zijn er ook andere mogelijkheden. Kunt u het antwoord nogmaals geven?",
"Kan de kraan afgesloten worden?",
"melding doorgeven aan de gekoppelde loodgieter",
"Wat is het probleem met het dak?",
"Veel voorkomende keuzes zijn bijvoorbeeld en verstopte dakgoot, lekke dakgoot of een kapot dakluik kranen of ventilatie maar natuurlijk zijn er ook andere mogelijkheden. Kunt u het antwoord nogmaals geven?",
"Dit is de verantwoordelijkheid van de bewoner",
"Bewoner doorverwijzen naar de eerstvolgende werkdag",
"Betreft het een gevaarlijke situatie?",
]

    language = "nl"

    # .........................Create directory for saving audio......................
    output_dir = "new_audio_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Starting real-time inference & playback...")
    t0 = time.time()

    # ..........................Launch ffplay subprocess for raw audio...................
    ffplay_cmd = [
        "ffplay",
        "-f", "f32le",        # 32-bit float PCM
        "-ar", "24000",       # sample rate
        "-ac", "1",           # mono
        "-nodisp",
        "-autoexit",
        "-loglevel", "quiet",
        "pipe:0"              # read from stdin
    ]
    proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

    for sentence in sentences:
        print(f"Processing sentence: {sentence[:50]}...")

        # ...............................Accumulate audio data for this sentence...............
        sentence_audio_data = b""

        # ..................................Stream inference from API for each sentence.........
        response = requests.post(f"{base_url}/inference_stream", params={"text": sentence, "language": language}, stream=True)
        if response.status_code != 200:
            print("Error in API request")
            return

        chunk_size = 4 * 24000 
        for chunk_bytes in response.iter_content(chunk_size=chunk_size):
            if chunk_bytes:
                proc.stdin.write(chunk_bytes)
                sentence_audio_data += chunk_bytes

        # ...........................Save sentence audio after processing....................
        if sentence_audio_data:
            # Convert to float32 
            audio_np = np.frombuffer(sentence_audio_data, dtype=np.float32)
        
            filename = sentence.replace(' ', '_').replace(',', '').replace('.', '').replace('?', '').replace('!', '') + '.wav'
            filepath = os.path.join(output_dir, filename)
            
            wavfile.write(filepath, 24000, audio_np)
            print(f"Audio saved as {filepath}.")
        
        

    # ...............Close stdin to signal ffplay to finish.............
    proc.stdin.close()
    proc.wait()
    print("Finished playback.")

if __name__ == "__main__":
    main()

# import time
# import subprocess
# import requests
# import numpy as np
# import os
# from scipy.io import wavfile
# import uuid  # For unique ID

# def main():
#     # API base URL
#     base_url = "http://127.0.0.1:8000"

#     # List of sentences to process in a loop
#     sentences = [
    
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#         "Het kan gebroken glas of een lekkage maar natuurlijk zijn er ook andere mogelijkheden",
#     ]
#     language = "nl"

#     # Create directory for saving audio
#     output_dir = "new_raw_outputs_1"
#     os.makedirs(output_dir, exist_ok=True)

#     print("Starting real-time inference & playback...")
#     t0 = time.time()

#     # Launch ffplay subprocess for raw audio (updated to 8kHz)
#     ffplay_cmd = [
#         "ffplay",
#         "-f", "f32le",        # 32-bit float PCM
#         "-ar", "24000",        # Updated sample rate to 8kHz
#         "-ac", "1",           # mono
#         "-nodisp",
#         "-autoexit",
#         "-loglevel", "quiet",
#         "pipe:0"              # read from stdin
#     ]
#     proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

#     for sentence in sentences:
#         print(f"Processing sentence: {sentence[:50]}...")

#         # Accumulate audio data for this sentence
#         sentence_audio_data = b""

#         # Fixed API call parameters
#         headers = {
#             'auth_key': 'jipFLutdqFIkNpKIupi5H9tAn9h2XI2OMAddFiPNR4E',  # Add your actual auth key
#             'Content-Type': 'application/json'
#         }

#         params = {
#             "text": sentence, 
#             "lang": language,  # Changed from "language" to "lang"
#             "model_name": "XTTS",  # Use proper model name
#             "text_type": "ai_system"  # Add required parameter
#         }

#         # Stream inference from API for each sentence
#         try:
#             response = requests.post(
#                 f"{base_url}/generate_audio_by_text_inference_stream", 
#                 params=params,
#                 headers=headers,
#                 stream=True
#             )
            
#             if response.status_code != 200:
#                 print(f"Error in API request: {response.status_code}")
#                 print(f"Response: {response.text}")
#                 continue
                
#         except Exception as e:
#             print(f"Error occurred: {str(e)}")
#             continue

#         chunk_size = 4 * 24000  # Updated chunk size for 8kHz
#         for chunk_bytes in response.iter_content(chunk_size=chunk_size):
#             if chunk_bytes:
#                 proc.stdin.write(chunk_bytes)
#                 sentence_audio_data += chunk_bytes

#         # Save sentence audio after processing
#         if sentence_audio_data:
#             try:
#                 # Convert to float32
#                 audio_np = np.frombuffer(sentence_audio_data, dtype=np.float32)
                
#                 # Generate unique filename with timestamp or UUID
#                 timestamp = str(int(time.time()))  # Unix timestamp
#                 unique_id = str(uuid.uuid4())[:8]  # Short UUID
#                 filename = f"{timestamp}_{unique_id}.wav"
#                 filepath = os.path.join(output_dir, filename)
                
#                 wavfile.write(filepath, 24000, audio_np)  # 24kHz
#                 print(f"Audio saved as {filepath}.")
                
#             except ValueError as e:
#                 print(f"Error converting audio data: {e}")
#                 print(f"Audio data length: {len(sentence_audio_data)} bytes")
#                 # Save raw data for debugging
#                 debug_filename = f"debug_{timestamp}_{unique_id}.bin"
#                 with open(os.path.join(output_dir, debug_filename), 'wb') as f:
#                     f.write(sentence_audio_data)
#                 print(f"Raw data saved as {debug_filename} for debugging")

#     # Close stdin to signal ffplay to finish
#     proc.stdin.close()
#     proc.wait()
#     print("Finished playback.")

# if __name__ == "__main__":
#     main()
