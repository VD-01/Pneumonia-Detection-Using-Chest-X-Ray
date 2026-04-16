import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_resnet_model
from tqdm import tqdm

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=15, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} processing"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def main():
    data_dir = r"C:\Users\vishn\Downloads\pnucxr\chest_xray"
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} not found.")
        print("Please ensure the chest_xray directory is located at exactly this path.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    num_workers = 4
    num_epochs = 15
    learning_rate = 0.001

    print("Loading data...")
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir, batch_size, num_workers)
    print(f"Classes: {class_names}")

    model = get_resnet_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    best_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=num_epochs, device=device)
    
    save_path = "best_model.pth"
    torch.save(best_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
