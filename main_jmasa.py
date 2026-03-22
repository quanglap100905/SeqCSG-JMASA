import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from torch.optim import AdamW
import transformers.models.bart.modeling_bart as bart_modeling

_original_expand_mask = bart_modeling._expand_mask
def _patched_expand_mask(mask, dtype, tgt_len=None):
    if mask.dim() == 4:
        return mask
    return _original_expand_mask(mask, dtype, tgt_len)
bart_modeling._expand_mask = _patched_expand_mask

from dataloader_jmasa import JMASADataset
from model_jmasa import JMASAModel
from utils_jmasa import train_jmasa_epoch, eval_jmasa_joint, Log, EarlyStopping

class Args:
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-5
    DATA_DIR = "/kaggle/working/hotel_data"
    IMG_DIR = "/kaggle/working/hotel_data/images"
    SAVE_PATH = "./log_jmasa_joint/best_jmasa_joint.pth"

def load_data(args):
    print("📂 Loading Joint Data...")
    train_path = os.path.join(args.DATA_DIR, "train_jmasa_joint.json")
    test_path = os.path.join(args.DATA_DIR, "test_jmasa_joint.json")
    
    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(train_data)} train | {len(test_data)} test samples.")
    return train_data, test_data

if __name__ == '__main__':
    args = Args()
    log_dir = './log_jmasa_joint'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = Log(log_dir, "jmasa_run").get_logger()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"⚙️ Running on: {device}")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    train_data, test_data = load_data(args)
    
    # DATALOADERS
    train_ds = JMASADataset(train_data, tokenizer, args.MAX_LEN, args.IMG_DIR)
    test_ds = JMASADataset(test_data, tokenizer, args.MAX_LEN, args.IMG_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=2)

    # MODEL & OPTIMIZER
    model = JMASAModel(tokenizer).to(device)
    optimizer = AdamW(model.parameters(), lr=args.LEARNING_RATE)
    
    # EarlyStopping
    early_stopping = EarlyStopping(patience=15, delta=0.000001, path=args.SAVE_PATH, trace_func=logger.info)

    logger.info("🚀 START JOINT TRAINING (MATE + MASC)...")
    
    for epoch in range(args.EPOCHS):
        logger.info(f"--- Epoch {epoch + 1}/{args.EPOCHS} ---")
        
        # 1. Train Joint
        avg_loss = train_jmasa_epoch(model, train_loader, optimizer, device)
        
        # 2. Eval Joint
        current_jmasa_f1 = eval_jmasa_joint(model, test_loader, device)
        
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | JMASA F1: {current_jmasa_f1:.4f}")
        
        early_stopping(current_jmasa_f1, model, epoch, optimizer)
        
        if early_stopping.early_stop:
            logger.info("🛑 Early stopping triggered!")
            break

    logger.info("✅ ALL TASKS COMPLETED. Best model saved.")
