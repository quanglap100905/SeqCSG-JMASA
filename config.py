import os

class Config:
    # ROOT FILE
    RAW_JSON = '/kaggle/input/datasets/tisdang/hotel-review-splitted/text_image_dataset.json'
    TRIPLE_CSV = '/kaggle/input/datasets/quanglapnguyen/top5-triples/top5_triples.csv'
    IMG_DIR = '/kaggle/working/hotel_data/images'
    
    # PROCESSED DATA
    PROCESSED_DIR = './data/processed'
    TRAIN_JSON = os.path.join(PROCESSED_DIR, 'train_jmasa_joint.json')
    TEST_JSON = os.path.join(PROCESSED_DIR, 'test_jmasa_joint.json')
    
    # OUTPUT
    SAVE_DIR = './output'
    CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
    
    # HYPERPARAMETER
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 2e-5
    NEGATIVE_RATIO = 0.8
    NUM_WORKERS = 2
