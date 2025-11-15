import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset import test_dataset
import pandas as pd
from args import get_args


def evaluate_model(model, model_name,device):
    args = get_args()
    
    # Load test CSV
    df_test = pd.read_csv(os.path.join(args.csv_dir, 'test.csv'))

    # create dataset
    test_df = test_dataset(df_test)
    # create loader
    test_loader = DataLoader(test_df, batch_size=1, shuffle=False)

    # Output directory
    out_dir = os.path.join(args.out_dir, model_name.replace(".pth", ""))
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img = batch["image"].to(device)
            path_img = df_test["xrays"].iloc[i]  
            base_name = os.path.splitext(os.path.basename(path_img))[0]

            # Forward pass
            logits = model(img)                        # (1,1,H,W)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

            # Convert back since openCV does not like writing other formats...
            pred_np = pred.cpu().numpy()[0, 0] * 255
            pred_np = pred_np.astype(np.uint8)

            out_path = os.path.join(out_dir, f"{base_name}_pred.png")
            cv2.imwrite(out_path, pred_np)



