import torch
import torch.nn as nn
from transformers import BartModel
import torchvision.models as models

class JMASAOutput:
    def __init__(self, loss, mate_logits, masc_logits):
        self.loss = loss
        self.mate_logits = mate_logits
        self.masc_logits = masc_logits

class JMASAModel(nn.Module):
    def __init__(self, tokenizer):
        super(JMASAModel, self).__init__()
        self.tokenizer = tokenizer
        self.bart = BartModel.from_pretrained("facebook/bart-base")
        
        # Visual Branch (ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_proj = nn.Linear(2048, 768)
        self.dropout = nn.Dropout(0.1)

        self.mate_classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 7) 
        )
        
        self.masc_classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 3)
        )

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, mate_labels=None, masc_labels=None, images=None):
        # 1. ENCODER & VISUAL FUSION
        if attention_mask.dim() == 3: 
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            extended_attention_mask = attention_mask
            
        enc_out = self.bart.encoder(input_ids=input_ids, attention_mask=extended_attention_mask)
        text_graph_feats = enc_out.last_hidden_state 

        # Process image
        img_feats = self.resnet(images).reshape(images.size(0), -1)
        img_embeds = self.visual_proj(img_feats).unsqueeze(1)
        img_embeds = self.dropout(img_embeds)
        
        # Fusion (Image + Text/Graph)
        fused_embeds = torch.cat((img_embeds, text_graph_feats), dim=1)
        
        # Mask for Fusion
        pad_token_id = self.bart.config.pad_token_id
        text_padding_mask = (input_ids != pad_token_id).long() 
        img_mask = torch.ones(images.size(0), 1, device=images.device).long()
        fused_mask = torch.cat((img_mask, text_padding_mask), dim=1) 

        # 2. DECODER
        dec_out = self.bart.decoder(
            input_ids=decoder_input_ids, 
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_embeds,
            encoder_attention_mask=fused_mask
        )
        dec_feats = dec_out.last_hidden_state

        # 3. FEATURE EXTRACTION 
        h_shared = torch.mean(dec_feats, dim=1) 

        # 4. PREDICT
        mate_logits = self.mate_classifier(h_shared)
        masc_logits = self.masc_classifier(h_shared)

        # 5. LOSS CALCULATION
        loss = None
        if mate_labels is not None and masc_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_mate = loss_fct(mate_logits, mate_labels)
            
            loss_fct_masc = nn.CrossEntropyLoss(ignore_index=-100)
            loss_masc = loss_fct_masc(masc_logits, masc_labels)
            
            loss = loss_mate + loss_masc

        return JMASAOutput(loss=loss, mate_logits=mate_logits, masc_logits=masc_logits)
