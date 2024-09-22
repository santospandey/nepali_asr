import time
import editdistance as ed
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime

from model.configs import (
    INPUT_DIM,
    MODEL_NAME,
    NUM_UNQ_CHARS,
    SR,
    UNQ_CHARS,
    device_name,
)

from model.model import get_model
from model.utils import (
    CER_from_mfccs,
    batchify,
    clean_single_wav,
    gen_mfcc,
    indices_from_texts,
    load_model,
)

def train_model(model, optimizer, train_data, test_data, epochs=100, batch_size=50):
    train_losses, test_losses, test_CERs = [], [], []
    current_CER = 999999999999999

    with tf.device(device_name):
        for e in range(epochs):
            start_time = time.time()
            train_loss, test_loss, test_CER = 0, 0, 0

            print(f"Training epoch: {e + 1}")
            train_loss, train_batch_count = run_epoch(
                model, optimizer, train_data, batch_size, training=True
            )

            print(f"Testing epoch: {e + 1}")
            test_loss, test_CER, test_batch_count = run_epoch(
                model, optimizer, test_data, batch_size, training=False
            )

            # Average losses and CER
            train_loss /= train_batch_count
            test_loss /= test_batch_count
            test_CER /= test_batch_count

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_CERs.append(test_CER)

            rec = (
                f"Epoch: {e + 1}, Train Loss: {train_loss:.2f}, "
                f"Test Loss: {test_loss:.2f}, Test CER: {test_CER * 100:.2f}% "
                f"in {time.time() - start_time:.2f} secs."
            )
            print(rec)
            if test_CER < current_CER:
                # Save the trained model with a timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_save_path = f"model/asr_model_{test_CER}_{timestamp}.h5"
                model.save(model_save_path)
                current_CER = test_CER

    print(f"Model saved to {model_save_path} \u2705")

    plot_metrics(train_losses, test_losses, test_CERs, epochs)


def run_epoch(model, optimizer, data, batch_size, training=True):
    loss = 0
    CER = 0
    batch_count = 0

    for start in tqdm(range(0, len(data[0]), batch_size)):
        end = min(start + batch_size, len(data[0]))
        x, target, target_lengths, output_lengths = batchify(
            data[0][start:end], data[1][start:end], UNQ_CHARS
        )

        if training:
            with tf.GradientTape() as tape:
                output = model(x, training=True)
                loss_value = K.ctc_batch_cost(target, output, output_lengths, target_lengths)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        else:
            output = model(x, training=False)
            loss_value = K.ctc_batch_cost(target, output, output_lengths, target_lengths)

        loss += np.average(loss_value.numpy())
        batch_count += 1

        if not training:
            input_len = np.ones(output.shape[0]) * output.shape[1]
            decoded_indices = K.ctc_decode(output, input_length=input_len, greedy=False, beam_width=100)[0][0]
            target_indices = [sent[sent != 0].tolist() for sent in target]
            predicted_indices = [sent[sent > 1].numpy().tolist() for sent in decoded_indices]

            for pred, truth in zip(predicted_indices, target_indices):
                ed_dist = ed.distance(pred, truth)
                CER += ed_dist / len(truth) if truth else 0

    return loss, CER, batch_count


def plot_metrics(train_losses, test_losses, test_CERs, epochs):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_CERs, label="Test CER")
    plt.xlabel("Epochs")
    plt.ylabel("CER")
    plt.title("Character Error Rate (CER)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def load_data(wavs_dir, texts_dir):
    texts_df = pd.read_csv(texts_dir, sep="\t", header=None, names=["file", "speaker", "text"])
    train_wavs = []
    for f_name in texts_df["file"]:
        try:
            wav, _ = librosa.load(f"{wavs_dir}/{f_name}.flac", sr=SR)
            train_wavs.append(wav)
            # print(f"{wavs_dir}/{f_name} found")
        except FileNotFoundError:
            pass
            # print(f"Warning: Audio file '{wavs_dir}/{f_name}' not found.")

    train_texts = texts_df["text"].tolist()
    return train_wavs, train_texts


if __name__ == "__main__":
    model = get_model(
        INPUT_DIM,
        NUM_UNQ_CHARS,
        num_res_blocks=5,
        num_cnn_layers=2,
        cnn_filters=50,
        cnn_kernel_size=15,
        rnn_dim=170,
        rnn_dropout=0.15,
        num_rnn_layers=2,
        num_dense_layers=1,
        dense_dim=340,
        model_name=MODEL_NAME,
        rnn_type="lstm",
        use_birnn=True,
    )
    print("Model defined \u2705 \u2705 \u2705 \u2705\n")

    optimizer = tf.keras.optimizers.Adam()

    print("Loading data.....")
    train_wavs, train_texts = load_data("download/wavs", "download/transcripts/utt_spk_text.tsv")
    print("Data loaded \u2705 \u2705 \u2705 \u2705\n")

    print("Cleaning the audio files.....")
    train_wavs = [clean_single_wav(wav) for wav in train_wavs]
    print("Audio files cleaned \u2705 \u2705 \u2705 \u2705\n")

    print("Generating mfcc features.....")
    train_wavs = [gen_mfcc(wav) for wav in train_wavs]
    print("MFCC features generated \u2705 \u2705 \u2705 \u2705\n")

    train_wavs, test_wavs, train_texts, test_texts = train_test_split(
        train_wavs, train_texts, test_size=0.1
    )

    train_model(model, optimizer, (train_wavs, train_texts), (test_wavs, test_texts), epochs=60, batch_size=100)

    # Save the trained model with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"model/asr_model_{timestamp}.h5"
    model.save(model_save_path)

    print(f"Model saved to {model_save_path} \u2705")
