import torch, os, numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class JMASADataset(Dataset):
    def __init__(self, data, tokenizer, max_len, image_dir):
        self.data, self.tokenizer, self.max_len, self.image_dir = data, tokenizer, max_len, image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, item):
        entry = self.data[item]
        review = entry['review_text']
        chunk = entry['candidate_chunk']
        caption = entry['caption']
        triples_data = entry['triples']
        img_id = entry['image_id']
        
        mate_label = entry.get('mate_label', 6)
        masc_label = entry.get('masc_label', -100)

        # 1. TOKENIZE & BUILD INPUT IDS
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        
        # S u C (Caption + Review)
        context_text = f"{caption}. {review}"
        ids_context = self.tokenizer.encode(context_text, add_special_tokens=False)
        input_ids = [bos] + ids_context + [eos]
        range_context = (0, len(input_ids)) 
        
        # Triples
        range_triples = []
        for t in triples_data:
            t_ids = self.tokenizer.encode(t['text'], add_special_tokens=False)
            start = len(input_ids)
            input_ids.extend(t_ids + [eos])
            range_triples.append((start, len(input_ids)))

        # Truncate
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            range_context = (0, min(range_context[1], self.max_len))
            range_triples = [(s, min(e, self.max_len)) for s, e in range_triples if s < self.max_len]

        # Padding
        real_len = len(input_ids)
        padding_len = self.max_len - real_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        
        # 2. BUILD VISIBLE MATRIX 
        visible_matrix = np.full((self.max_len, self.max_len), -1e9, dtype=np.float32)
        
        def set_visible(r1, r2):
            visible_matrix[r1[0]:r1[1], r2[0]:r2[1]] = 0.0

        # Context see All and All see Context
        set_visible(range_context, (0, self.max_len)) 
        set_visible((0, self.max_len), range_context)
        
        # Triple can see itself
        for r in range_triples:
            set_visible(r, r)
            
        # Connect triples with same Entity (sub/obj)
        for i in range(len(range_triples)):
            for j in range(i + 1, len(range_triples)):
                ents_i = {str(triples_data[i]['sub']).lower(), str(triples_data[i]['obj']).lower()}
                ents_j = {str(triples_data[j]['sub']).lower(), str(triples_data[j]['obj']).lower()}
                
                if not ents_i.isdisjoint(ents_j):
                    set_visible(range_triples[i], range_triples[j])
                    set_visible(range_triples[j], range_triples[i])
                    
        # Masking Padding & Diagonal
        visible_matrix[real_len:, :] = -1e9
        visible_matrix[:, real_len:] = -1e9
        np.fill_diagonal(visible_matrix, 0.0)

        # 3. DECODER
        decoder_text = f"Aspect {chunk} is <mask" 
        dec = self.tokenizer(decoder_text, max_length=32, padding='max_length', truncation=True, return_tensors='pt')

        # 4. IMAGE
        try:
            image = Image.open(os.path.join(self.image_dir, img_id)).convert("RGB")
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(visible_matrix, dtype=torch.float),
            'decoder_input_ids': dec['input_ids'].flatten(),
            'decoder_attention_mask': dec['attention_mask'].flatten(),
            'mate_labels': torch.tensor(mate_label, dtype=torch.long),
            'masc_labels': torch.tensor(masc_label, dtype=torch.long),
            'image_pixels': image
        }
