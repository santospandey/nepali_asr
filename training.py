import time

import editdistance as ed  # or edit_distance if using that library
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
    strategy,
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


def train_model(
    model,
    optimizer,
    train_wavs,
    train_texts,
    test_wavs,
    test_texts,
    epochs=100,
    batch_size=50,
):
    # Lists to store the losses for visualization
    train_losses = []
    test_losses = []
    test_CERs = []

    with strategy.scope():
        for e in range(0, epochs):
            start_time = time.time()
            len_train = len(train_wavs)
            len_test = len(test_wavs)
            train_loss = 0
            test_loss = 0
            test_CER = 0
            train_batch_count = 0
            test_batch_count = 0

            print("Training epoch: {}".format(e + 1))
            for start in tqdm(range(0, len_train, batch_size)):

                end = None
                if start + batch_size < len_train:
                    end = start + batch_size
                else:
                    end = len_train
                x, target, target_lengths, output_lengths = batchify(
                    train_wavs[start:end], train_texts[start:end], UNQ_CHARS
                )

                with tf.GradientTape() as tape:
                    output = model(x, training=True)

                    loss = K.ctc_batch_cost(
                        target, output, output_lengths, target_lengths
                    )

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                train_loss += np.average(loss.numpy())
                train_batch_count += 1

            print("Testing epoch: {}".format(e + 1))
            for start in tqdm(range(0, len_test, batch_size)):

                end = None
                if start + batch_size < len_test:
                    end = start + batch_size
                else:
                    end = len_test
                x, target, target_lengths, output_lengths = batchify(
                    test_wavs[start:end], test_texts[start:end], UNQ_CHARS
                )

                output = model(x, training=False)

                # Calculate CTC Loss
                loss = K.ctc_batch_cost(target, output, output_lengths, target_lengths)

                test_loss += np.average(loss.numpy())
                test_batch_count += 1

                """
                    The line of codes below is for computing evaluation metric (CER) on internal validation data.
                """
                input_len = np.ones(output.shape[0]) * output.shape[1]
                decoded_indices = K.ctc_decode(
                    output, input_length=input_len, greedy=False, beam_width=100
                )[0][0]

                # Remove the padding token from batchified target texts
                target_indices = [sent[sent != 0].tolist() for sent in target]

                # Remove the padding, unknown token, and blank token from predicted texts
                predicted_indices = [
                    sent[sent > 1].numpy().tolist() for sent in decoded_indices
                ]  # idx 0: padding token, idx 1: unknown, idx -1: blank token

                len_batch = end - start
                for i in range(len_batch):
                    pred = predicted_indices[i]
                    truth = target_indices[i]
                    ed_dist = ed.eval(pred, truth)
                    test_CER += ed_dist / len(truth)
                test_CER /= len_batch

            train_loss /= train_batch_count
            test_loss /= test_batch_count
            test_CER /= test_batch_count

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_CERs.append(test_CER)

            rec = "Epoch: {}, Train Loss: {:.2f}, Test Loss {:.2f}, Test CER {:.2f} % in {:.2f} secs.\n".format(
                e + 1, train_loss, test_loss, test_CER * 100, time.time() - start_time
            )

            print(rec)

            # Save the trained model with a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = f"/kaggle/working/model/asr_model_{e}_{timestamp}.h5"
            model.save(model_save_path)


def load_data(wavs_dir, texts_dir, reduction_factor=5):
    texts_df = pd.read_csv(texts_dir, sep="\t", header=None, names=["file", "speaker", "text"])
    train_wavs = []

    for f_name in texts_df["file"]:
        try:
            wav, _ = librosa.load(f"{wavs_dir}/{f_name}.flac", sr=SR)
            train_wavs.append(wav)
        except FileNotFoundError:
            pass

    train_texts = texts_df["text"].tolist()

    # Randomly select 1/5th of the data
    indices = np.random.choice(len(train_wavs), size=len(train_wavs) // reduction_factor, replace=False)
    reduced_train_wavs = [train_wavs[i] for i in indices]
    reduced_train_texts = [train_texts[i] for i in indices]
    return reduced_train_wavs, reduced_train_texts



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
    train_wavs, train_texts = load_data("download/wavs", "download/transcripts/utt_spk_text.tsv", reduction_factor=25)
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

    train_model(model, optimizer, train_wavs, train_texts, test_wavs, test_texts, epochs=30, batch_size=50)
