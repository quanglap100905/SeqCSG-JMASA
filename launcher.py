import os
import sys
import time

WORK_DIR = "/kaggle/working/seqcsg-hotel/SeqCSG-Hotel"
DATA_DIR = "/kaggle/working/hotel_data"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 1. ENVIRONMENT
try:
    import transformers
    import torch
    import spacy
    libs = "transformers==4.26.1 tokenizers==0.13.2 protobuf==3.20.3 accelerate==0.24.1 --quiet"
    os.system(f"pip install {libs}")
    os.system("python -m spacy download en_core_web_sm --quiet")
except ImportError:
    os.system("pip install transformers==4.26.1 tokenizers==0.13.2 --quiet")

# 2. PREPARE DATA
cmd = "python -u prepare_jmasa.py"
exit_code = os.system(cmd)
if exit_code != 0:
  print(f"\n❌ ERROR: Code {exit_code}")
else:
  print("Data is ready")
  
# 3. TRAINING
cmd = "python -u main_jmasa.py"
exit_code = os.system(cmd)
if exit_code != 0:
  print(f"\n❌ ERROR: Code {exit_code}")
else:
  print(f"💾 Checkpoint: {WORK_DIR}/log_jmasa_graph/best_model_jmasa.pth")
