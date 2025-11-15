#! /usr/bin/python3
import time
import numpy as np
import torch
from model import UNetLext
from args import get_args
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from dataset import xray_dataset
from trainer import train_model
from evaluate import evaluate_model


def main():
    args = get_args()

    ## Step 1: 
    train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
    val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

    # Step 2: preparing dataset
    train_dataset = xray_dataset(train_set)
    val_dataset = xray_dataset(val_set)

    #Step 3 initialize dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    # Initializing the model
    model = UNetLext(
        input_channels=1,
        output_channels=1,
        pretainer=False,
        path_pretrained="",
        restore_weights=False,
        path_weights=""
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training
    train_model(model, train_loader, val_loader, device)

    #save model with current time in name
    timestring = time.strftime("%Y%m%d - %H%M")
    filename = f"{timestring}.pth"
    filepath = os.path.join(args.out_dir, filename)
    torch.save(model.state_dict(), filepath)

    evaluate_model(
    model=model,
    model_name=filename,
    device = device   
    )







if __name__ == "__main__":
    main()

