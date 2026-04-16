import os
import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import get_resnet_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

def evaluate_model(model, dataloader, class_names, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix on Test Data')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

def main():
    data_dir = r"C:\Users\vishn\Downloads\pnucxr\chest_xray"
    model_path = "best_model.pth"
    
    if not os.path.exists(data_dir):
         print(f"Error: Dataset directory {data_dir} not found.")
         return
         
    if not os.path.exists(model_path):
        print(f"Error: Model weights {model_path} not found. Please run train.py first.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading test data...")
    dataloaders, _, class_names = get_dataloaders(data_dir, batch_size=32, num_workers=4)
    test_loader = dataloaders['test']

    print("Loading model...")
    model = get_resnet_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)

    print("Starting evaluation...")
    evaluate_model(model, test_loader, class_names, device)

if __name__ == '__main__':
    main()
