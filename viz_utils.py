import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_attention_map(attentions, layer_idx=-1, head_idx=None):
    """
    Extracts attention matrix from model outputs.
    
    Args:
        attentions: shape (layer, heads, seq, seq)
        layer_idx (int): Which layer to visualize.
        head_idx (int): Which head to visualize. If None, averages all heads.
    """
    # attns shape: (batch, heads, seq, seq)
    attns = attentions[layer_idx]
    
    # Take first item in batch
    attn_matrix = attns[0].detach().cpu().float()
    
    if head_idx is not None:
        attn_matrix = attn_matrix[head_idx]
    else:
        # Average across heads
        attn_matrix = attn_matrix.mean(dim=0)
        
    return attn_matrix

def plot_attention_heatmap(attn_matrix, tokens=None, title="Attention Map"):
    """
    Plots the attention heatmap.
    """
    plt.figure(figsize=(10, 8))
    
    # Log scale is often better for attention visualization 
    # as sink tokens dominate linear scales
    log_attn = torch.log(attn_matrix + 1e-9).numpy()
    
    sns.heatmap(log_attn, cmap="viridis", square=True)
    
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    # If tokens are provided, we can try to label axes (optional, can be crowded)
    if tokens is not None and len(tokens) < 50:
        plt.xticks(np.arange(len(tokens)) + 0.5, tokens, rotation=90, fontsize=8)
        plt.yticks(np.arange(len(tokens)) + 0.5, tokens, rotation=0, fontsize=8)
        
    plt.show()

def highlight_image_regions(input_ids, vision_start_id, vision_end_id):
    """
    Helper to find start/end indices of images for visualization overlays.
    """
    seq = input_ids[0].tolist()
    regions = []
    start = -1
    for i, token in enumerate(seq):
        if token == vision_start_id:
            start = i
        elif token == vision_end_id and start != -1:
            regions.append((start, i))
            start = -1
    return regions