import tensorflow as tf


# Necessary global configs
LOAD_MFCC_FILES = True

FRAME_SIZE = 160
SR = 16000
FRAME_RATE = int( SR / FRAME_SIZE )
N_MFCC = 13
HOP_LENGTH = 40

assert FRAME_SIZE % HOP_LENGTH == 0

INPUT_DIM = int(N_MFCC * (FRAME_SIZE / HOP_LENGTH))

UNQ_CHARS = [' ', 'ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ॠ', '\u200c', '\u200d', '।']
UNQ_CHARS = ['0', 'u' ] + sorted(UNQ_CHARS) + ['-'] #"0" -> padding char,"u" -> unknown chars "-" -> blank char
NUM_UNQ_CHARS = len(UNQ_CHARS) # +1 is for '-' blank at last

MODEL_NAME = "ASR_model"

# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print(f"Using TPU: {tpu}")
else:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy()  # Use GPU
        print(f"Using GPU: {gpus}")
    else:
        strategy = tf.distribute.get_strategy()  # Default to CPU
        print("Using CPU")
