import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from extractor_sd import load_model, process_features_and_mask, get_mask
from utils.utils_correspondence import ft_pca, co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace
import matplotlib.pyplot as plt
import sys
from extractor_dino import ViTExtractor
from sklearn.decomposition import PCA as sklearnPCA
import math
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def compute_fuse_feature(model, aug, save_path, files, category, mask=False, dist='cos', real_size=960):
    if type(category) == str:
        category = [category]
    img_size = 840 if DINOV2 else 244
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
    
    model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if DINOV2 else 'key'
    stride = 14 if DINOV2 else 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
    
    input_text = "a photo of "+category[-1][0] if TEXT_INPUT else None

    current_save_results = 0

    N = len(files) // 2
    pbar = tqdm(total=N)
    result = []
    if 'Anno' in files[0]:
        Anno=True
    else:
        Anno=False
    for pair_idx in range(N):

        # Load image 1
        img1 = Image.open(files[2*pair_idx]).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        with torch.no_grad():
            
            features1 = process_features_and_mask(model, aug, img1_input, input_text=input_text,  mask=False, raw=True)
            processed_features1 = ft_pca(features1, PCA_DIMS)
            img1_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
            img1_batch = extractor.preprocess_pil(img1)
            img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)

            # normalize the features
            img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
            img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
            img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
        
    return result
