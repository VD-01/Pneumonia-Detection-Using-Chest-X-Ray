import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import get_resnet_model

def predict_image(image_path, model_path="best_model.pth", class_names=['NORMAL', 'PNEUMONIA']):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Load model
    try:
        model = get_resnet_model(num_classes=len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        conf, predicted_class = torch.max(probabilities, 0)
        
    class_idx = predicted_class.item()
    class_name = class_names[class_idx]
    confidence = conf.item() * 100
    
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Pneumonia from Chest X-Ray")
    parser.add_argument("image_path", type=str, help="Path to the chest X-ray image (e.g., .jpeg or .png)")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the trained model weights")
    
    args = parser.parse_args()
    predict_image(args.image_path, args.model_path)
