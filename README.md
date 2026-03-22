# 🏨 SeqCSG-JMASA: Joint Multimodal Aspect Sentiment Analysis

This repository implements a **JMASA** (Joint Multimodal Aspect Sentiment Analysis) model based on the **BART-base** architecture. The model integrates textual reviews and visual knowledge (**Image Triples**) through a **Visible Matrix (Graph Matrix)** mechanism to optimize extraction performance for hotel reviews.

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/quanglap100905/SeqCSG-JMASA.git
cd SeqCSG-JMASA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the SpaCy language model
python -m spacy download en_core_web_sm

# 4. Run
python prepare_data.py
python train.py
