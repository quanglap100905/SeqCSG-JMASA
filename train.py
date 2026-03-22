import torch
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

from config import Config
from models.dataloader_jmasa import JMASADataset
from models.model_jmasa import JMASAModel
from utils.utils_jmasa import train_jmasa_epoch, eval_jmasa_joint, Log, EarlyStopping
import json

def main():
    if not os.path.exists(Config.SAVE_DIR): os.makedirs(Config.SAVE_DIR)
    logger = Log(Config.SAVE_DIR, Config.LOG_NAME).get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    with open(Config.TRAIN_JSON) as f: train_data = json.load(f)
    with open(Config.TEST_JSON) as f: test_data = json.load(f)

    train_loader = DataLoader(JMASADataset(train_data, tokenizer, Config.MAX_LEN, Config.IMG_DIR), 
                              batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(JMASADataset(test_data, tokenizer, Config.MAX_LEN, Config.IMG_DIR), 
                             batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, pin_memory=True)

    model = JMASAModel(tokenizer).to(device)
    optimizer = AdamW(model.parameters(), lr=Config.LR)
    early_stopping = EarlyStopping(patience=7, delta=0.000001, path=Config.CHECKPOINT_PATH, trace_func=logger.info)

    for epoch in range(Config.EPOCHS):
        loss = train_jmasa_epoch(model, train_loader, optimizer, device)
        f1 = eval_jmasa_joint(model, test_loader, device)
        logger.info(f"Epoch {epoch+1} | Loss: {loss:.4f} | JMASA F1: {f1:.4f}")
        early_stopping(f1, model, epoch, optimizer)
        if early_stopping.early_stop: break

if __name__ == "__main__": main()
