TEST_N_AGENTS = 8

NODE_INPUT_DIM = 5
OTHER_INFO_INPUT_DIM = 3
EMBEDDING_DIM = 64
K_SIZE = 25  # the number of neighbors

USE_GPU = True  # do you want to use GPUS?
NUM_GPU = 1
NUM_META_AGENT = 24  # the number of processes
FOLDER_NAME = 'clean'
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs'

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = False  # do you want to save GIFs
