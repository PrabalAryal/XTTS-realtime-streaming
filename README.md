To run this project follow the steps below 
  - setup a python environment (3.10)
  - run the following command in the terminal
    - wget https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json
    - wget https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth
    - wget https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json
    - pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    - pip install transformers==4.32.0
    - pip install gruut[de,es,fr]==2.4.0
    - pip install protobuf==3.20.3 
    - pip install sounddevice
  - add the the downloaded model form the first 3 command into a directory named xtts_pretrained
  - run the two files men.py and infe.py inside model directory ( python men.py and python infe.py)



 
