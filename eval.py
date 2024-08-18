# You can find all of the necessary functions properly documented in utils.py
import os
import csv
import random
from model.configs import UNQ_CHARS
from model.utils import CER_from_wavs, ctc_softmax_output_from_wavs, load_model, load_wav, plot_losses, predict_from_wavs


if __name__ == "__main__":

    # Loads the trained model
    print("Loading model.....")
    model = load_model("model/asr_model.h5")
    print("Model loaded \u2705 \u2705 \u2705 \u2705\n")
    

    # Loads wav file
    wavs = []
    print("Loading wav files.....")
    basedir = "datasets/wavs"
    files = os.listdir(basedir)
    TOTAL_DATASIZE = len(files)
    TOTAL_SAMPLES = 10
    
    random_numbers = [random.randint(0, TOTAL_DATASIZE) for i in range(TOTAL_SAMPLES) ]
    
    wavs = [load_wav(os.path.join(basedir, files[i])) for i in random_numbers]
    
    """Gives the array of predicted sentences"""
    print("Predicting sentences.....")
    sentences, char_indices = predict_from_wavs(model, wavs, UNQ_CHARS)
    
    # Specify the file name
    csv_file = "datasets/transcripts/speaker.csv"
    identifier_to_text = {}

    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:            
            identifier_to_text[row[0]] = row[2]  # Map identifier to corresponding text

    labels = []
    files = [files[i] for i in random_numbers]
    for file in files:
        file = file.replace('.wav', '')
        if file in identifier_to_text:
            labels.append(identifier_to_text[file])
        else:
            print(f"{file} not found")
    
    print(".........................................................................\n")
    for i in range(len(labels)):
        print(f"{labels[i]} ==> {sentences[i]}" )
    print(".........................................................................\n")
    

    """Gives Character Error Rate (CER) between the targeted and predicted output"""
    print("Calculating CER.....")
    cer = CER_from_wavs(model, wavs, labels, UNQ_CHARS)
    print(cer, "\n")