from torchvision.models import resnet18, resnet50
import torch.nn as nn
import torch



# Positional Encoding
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=100, width=100):
        super().__init__()
        self.row_embed = nn.Embedding(height, d_model // 2)
        self.col_embed = nn.Embedding(width, d_model // 2)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, H, W, device):
        i = torch.arange(W, device=device)
        j = torch.arange(H, device=device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1)
        ], dim=-1)
        return pos.flatten(0, 1)

# DETR Model
class DETR(nn.Module):
    def __init__(self, num_classes=91, num_queries=100, hidden_dim=256):
    # def __init__(self, num_classes=91, num_queries=20, hidden_dim=256):
        super().__init__()
        # self.backbone = resnet50(pretrained=True)
        self.backbone = resnet18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.pos_encoding = PositionalEncoding2D(hidden_dim)
        # self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        features = self.backbone(x)
        H, W = features.shape[-2:]
        src = self.input_proj(features).flatten(2).permute(2, 0, 1)
        pos = self.pos_encoding(H, W, x.device).unsqueeze(1).repeat(1, B, 1)
        src = src + pos

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = query_embed  # Use query embeddings as the initial input

        memory = self.transformer.encoder(src)
        hs = self.transformer.decoder(tgt, memory).transpose(0, 1)

        class_logits = self.class_embed(hs)
        bboxes = self.bbox_embed(hs)
        return {"pred_logits": class_logits, "pred_boxes": bboxes}



