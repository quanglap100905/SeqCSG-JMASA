import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

class EarlyStopping:
    def __init__(self, patience=7, delta=0.0001, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, model, epoch, optimizer):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(current_score, model, epoch, optimizer)
        elif current_score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(current_score, model, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, score, model, epoch, optimizer):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': score
        }
        torch.save(checkpoint, self.path)
        self.trace_func(f'🚀 JMASA F1 Improved ({score:.4f}). Saving model to {self.path}')

class Log:
    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
    def get_logger(self): return self.logger

def train_jmasa_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training Joint"):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            decoder_input_ids=batch['decoder_input_ids'].to(device),
            decoder_attention_mask=batch['decoder_attention_mask'].to(device),
            mate_labels=batch['mate_labels'].to(device),
            masc_labels=batch['masc_labels'].to(device),
            images=batch['image_pixels'].to(device)
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_jmasa_joint(model, loader, device):
    model.eval()
    mate_preds, mate_gts = [], []
    jmasa_preds_list, jmasa_gts_list = [], []
    
    sent_map = {0: "Positive", 1: "Neutral", 2: "Negative"}
    cat_map = {
        0: "Facility", 1: "Amenity", 2: "Service", 
        3: "Branding", 4: "Experience", 5: "Loyalty", 6: "NOT_HOTEL"
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval Joint"):
            m_labels = batch['mate_labels'].to(device)
            s_labels = batch['masc_labels'].to(device)
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device),
                decoder_attention_mask=batch['decoder_attention_mask'].to(device),
                mate_labels=None, masc_labels=None,
                images=batch['image_pixels'].to(device)
            )
            
            m_pred = outputs.mate_logits.argmax(dim=-1).cpu().numpy()
            s_pred = outputs.masc_logits.argmax(dim=-1).cpu().numpy()
            m_gt = m_labels.cpu().numpy()
            s_gt = s_labels.cpu().numpy()

            mate_preds.extend(m_pred)
            mate_gts.extend(m_gt)

            for i in range(len(m_gt)):
                if m_gt[i] != 6 or m_pred[i] != 6:
                    gt_label = f"{cat_map[m_gt[i]]}:{sent_map.get(s_gt[i], 'Neutral')}" if m_gt[i] != 6 else "None:None"
                    pred_label = f"{cat_map[m_pred[i]]}:{sent_map[s_pred[i]]}" if m_pred[i] != 6 else "None:None"
                    
                    jmasa_gts_list.append(gt_label)
                    jmasa_preds_list.append(pred_label)

    # 1. MATE Macro F1
    mate_f1 = f1_score(mate_gts, mate_preds, average='macro', zero_division=0)
    
    # 2. JMASA Micro F1 
    jmasa_f1 = f1_score(jmasa_gts_list, jmasa_preds_list, average='micro', zero_division=0)

    print(f"\n" + "="*50)
    print(f"📊 JOINT REPORT")
    print(f"-"*50)
    print(f"MATE Macro F1: {mate_f1:.4f}")
    print(f"JMASA Joint F1: {jmasa_f1:.4f}")
    print(f"="*50)
    
    target_names_mate = ["Facility", "Amenity", "Service", "Branding", "Experience", "Loyalty", "NOT_HOTEL"]
    print("\n[MATE Detailed Report]")
    print(classification_report(mate_gts, mate_preds, labels=[0,1,2,3,4,5,6], target_names=target_names_mate, digits=4, zero_division=0))

    return jmasa_f1
