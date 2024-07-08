FOLDER_NAME = 'clean'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
SUMMARY_WINDOW = 32
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100

N_AGENTS = 4

CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0 # meter
DOWNSAMPLE_SIZE = NODE_RESOLUTION // CELL_SIZE

SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 1
FRONTIER_CELL_SIZE = 4 * CELL_SIZE

LOCAL_MAP_SIZE = 40  # meter
EXTENDED_LOCAL_MAP_SIZE = 6 * SENSOR_RANGE * 1.05

MAX_EPISODE_STEP = 128
REPLAY_SIZE = 40000
MINIMUM_BUFFER_SIZE = 10000
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 1

LOCAL_NODE_INPUT_DIM = 5
OTHER_INFO_INPUT_DIM = 3
EMBEDDING_DIM = 64

LOCAL_K_SIZE = 25  # the number of neighboring nodes
LOCAL_NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value

USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 8