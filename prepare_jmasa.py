import json, os, random, spacy, pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

ROOT_DIR = '/kaggle/working'
HOTEL_DATA_DIR = os.path.join(ROOT_DIR, 'hotel_data')
os.makedirs(HOTEL_DATA_DIR, exist_ok=True)

RAW_FILE = '/kaggle/input/datasets/tisdang/hotel-review-splitted/text_image_dataset.json'
TRIPLET_CSV = '/kaggle/input/datasets/quanglapnguyen/top5-triples/top5_triples.csv'
OUT_FILE = os.path.join(HOTEL_DATA_DIR, 'train_jmasa_joint.json')
IMG_DIR = '/kaggle/working/hotel_data/images' 

CAT_MAP = {"Facility": 0, "Amenity": 1, "Service": 2, "Branding": 3, "Experience": 4, "Loyalty": 5, "NOT_HOTEL": 6}
SENT_MAP = {"Positive": 0, "Neutral": 1, "Negative": 2}
NEGATIVE_RATIO = 0.8

nlp = spacy.load("en_core_web_sm")

def main():
    print("⚙️ START PREPARING DATA WITH TRIPLES & BALANCING...")

    # Load Triplets
    triplet_map = {}
    try:
        df_trips = pd.read_csv(TRIPLET_CSV)
        for _, row in df_trips.iterrows():
            img_name = str(row['image'])
            trip_data = {
                "text": f"{row['subject']} {row['relation']} {row['object']}",
                "sub": str(row['subject']).lower().strip(),
                "obj": str(row['object']).lower().strip()
            }
            if img_name not in triplet_map: triplet_map[img_name] = []
            if len(triplet_map[img_name]) < 5: 
                triplet_map[img_name].append(trip_data)
        print(f"✅ Loaded triplets for {len(triplet_map)} images.")
    except Exception as e:
        print(f"⚠️ Warning: Triplets not found. {e}")

    # Load Raw Data
    with open(RAW_FILE, 'r', encoding='utf-8') as f: 
        raw_data = json.load(f)
    
    positive_samples = []
    negative_samples = []

    for entry in tqdm(raw_data, desc="Processing"):
        img_id = entry.get('image_id')
        img_name = f"{img_id}.jpg"
        
        if not os.path.exists(os.path.join(IMG_DIR, img_name)):
            continue

        review = entry.get('review') or entry.get('review_text', "")
        caption = entry.get('photo_caption', "hotel view")
        triples = triplet_map.get(img_name, [])

        # Ground Truth Mapping
        gt_aspects = entry.get('review_aspects', [])
        gt_cats = entry.get('review_aspect_categories', [])
        gt_sents = entry.get('review_opinion_categories', [])
        
        mapping = {}
        for i, asp in enumerate(gt_aspects):
            term = asp.get('term', '').lower().strip() if isinstance(asp, dict) else str(asp).lower().strip()
            cat = gt_cats[i] if i < len(gt_cats) else "NOT_HOTEL"
            sent = gt_sents[i] if i < len(gt_sents) else "Neutral"
            mapping[term] = (cat, sent)

        # NLP Processing (Noun Chunks)
        doc = nlp(review)
        processed_chunks = set()
        for chunk in doc.noun_chunks:
            clean = chunk.text.lower().strip()
            for art in ["the ", "a ", "an "]:
                if clean.startswith(art): clean = clean[len(art):]
            
            if len(clean) < 2 or chunk.root.pos_ == "PRON": continue
            if clean in processed_chunks: continue
            processed_chunks.add(clean)

            res = mapping.get(clean, ("NOT_HOTEL", "Neutral"))
            cat_id = CAT_MAP.get(res[0], 6)
            sent_id = SENT_MAP.get(res[1], 1) if cat_id != 6 else -100

            sample = {
                "review_text": review,
                "candidate_chunk": clean,
                "mate_label": cat_id,
                "masc_label": sent_id,
                "image_id": img_name,
                "caption": caption,
                "triples": triples
            }

            if cat_id != 6: 
                positive_samples.append(sample)
            else: 
                negative_samples.append(sample)

    # BALANCING
    print(f"\n📊 Before balancing: Positive: {len(positive_samples)} | Negative: {len(negative_samples)}")
    
    num_neg_keep = int(len(positive_samples) * NEGATIVE_RATIO)
    random.seed(42)
    if len(negative_samples) > num_neg_keep:
        negative_samples = random.sample(negative_samples, num_neg_keep)
    
    all_samples = positive_samples + negative_samples
    
    # STRATIFIED SPLIT
    df = pd.DataFrame(all_samples)
    counts = df['mate_label'].value_counts()
    
    df['stratify_col'] = df['mate_label'].apply(lambda x: x if counts[x] > 1 else 6)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['stratify_col']
    )

    # To list dict
    train_list = train_df.drop(columns=['stratify_col']).to_dict(orient='records')
    test_list = test_df.drop(columns=['stratify_col']).to_dict(orient='records')

    with open(OUT_FILE, 'w') as f: json.dump(train_list, f, indent=4)
    with open(OUT_FILE.replace('train', 'test'), 'w') as f: json.dump(test_list, f, indent=4)
    
    print(f"🚀 FINISH DATA!")
    print(f"📊 Final Train: {len(train_list)} | Test: {len(test_list)}")
    print(f"📈 Category Distribution:\n{train_df['mate_label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()
