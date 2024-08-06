import os
import shutil
import zipfile

import pandas as pd
import requests

# URL of the dataset
dataset_url = "https://www.openslr.org/resources/43/ne_np_female.zip"
dataset_zip = "ne_np_female.zip"

# Download the dataset
print("Downloading dataset...")
response = requests.get(dataset_url, stream=True)
with open(dataset_zip, "wb") as f:
    shutil.copyfileobj(response.raw, f)
print("Download completed.")

# Unzip the dataset
print("Unzipping dataset...")
with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
    zip_ref.extractall("dataset")
print("Unzipping completed.")

# Create directories for audio files and transcripts
os.makedirs("datasets/wavs", exist_ok=True)
os.makedirs("datasets/transcripts", exist_ok=True)

# Move audio files and transcript files
for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.endswith(".wav"):
            shutil.move(os.path.join(root, file), "datasets/wavs/")
        elif file.endswith(".tsv"):
            shutil.move(os.path.join(root, file), "datasets/transcripts/")


# Load the CSV file into a DataFrame
df = pd.read_csv(
    "datasets/transcripts/line_index.tsv", sep="\t", header=None, names=["file", "text"]
)


# Define a function to extract speakerid from fileid
def extract_speakerid(fileid):
    return "_".join(fileid.split("_")[:2])


# Apply the function to create a new column 'speakerid'
df["speaker"] = df["file"].apply(extract_speakerid)

# Reorder and format columns
df = df[["file", "speaker", "text"]]

# Save the DataFrame to a new CSV file
df.to_csv("datasets/transcripts/speaker.csv", index=False, header=True)

# Clean up: remove the downloaded zip file and extracted folder
os.remove(dataset_zip)
shutil.rmtree("dataset")

print(
    "Dataset prepared. Audio files saved to 'wavs/' and transcripts saved to 'transcripts/'."
)
