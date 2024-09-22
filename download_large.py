import os
import requests
import zipfile
import shutil
from tqdm import tqdm

# Create directories for downloading files
download_dir = 'download'
transcript_dir = f"{download_dir}/transcripts"
temp_dir = f"{download_dir}/temp"
audio_dir = f"{download_dir}/wavs"

os.makedirs('download/wavs', exist_ok=True)
os.makedirs(transcript_dir, exist_ok=True)

# # List of zip files to download
# zip_files = [
#     "asr_nepali_0.zip",
#     "asr_nepali_1.zip",
#     "asr_nepali_2.zip",
#     "asr_nepali_3.zip",
#     "asr_nepali_4.zip",
#     "asr_nepali_5.zip",
#     "asr_nepali_6.zip",
#     "asr_nepali_7.zip",
#     "asr_nepali_8.zip",
#     "asr_nepali_9.zip",
#     "asr_nepali_a.zip",
#     "asr_nepali_b.zip",
#     "asr_nepali_c.zip",
#     "asr_nepali_d.zip",
#     "asr_nepali_e.zip",
#     "asr_nepali_f.zip"
# ]

# List of zip files to download
zip_files = [
    "asr_nepali_0.zip",
    "asr_nepali_1.zip",
    "asr_nepali_2.zip",
    "asr_nepali_3.zip",
    "asr_nepali_4.zip",
    "asr_nepali_5.zip"
]



# Base URL for the downloads
base_url = "https://openslr.org/resources/54/"

# Function to download and extract .zip files
def download_and_extract(zip_file):
    zip_path = os.path.join(download_dir, zip_file)
    
    # Download the zip file
    print(f"Downloading {zip_file}...")
    response = requests.get(base_url + zip_file, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        total_size = int(response.headers.get('content-length', 0))
        for data in tqdm(response.iter_content(chunk_size=4096), total=total_size // 4096, unit='KB'):
            f.write(data)

    # Extract .flac files from the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Walk through the source directory and find all .flac files
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.flac'):
                flac_file = os.path.join(root, file)
                # Copy the .flac file to the destination directory
                shutil.copy(flac_file, audio_dir)
                # print(f"Copied {flac_file} to {audio_dir}")

    print("All .flac files copied successfully!")

    shutil.copy(f"{temp_dir}/asr_nepali/utt_spk_text.tsv", transcript_dir)
    
    # Optionally remove the zip file after extraction
    os.remove(zip_path)
    shutil.rmtree(temp_dir)

# Download all zip files and extract .flac files
for zip_file in zip_files:
    download_and_extract(zip_file)

print("Download and extraction complete!")
