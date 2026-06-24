import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Class to capture activations (feature maps) and gradients
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate Grad-CAM
        # Pool the gradients across channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Multiply activations by pooled gradients
        for i in range(self.activations.shape[1]):
            self.activations[0, i, :, :] *= pooled_gradients[i]
            
        # Sum all channels and apply ReLU
        heatmap = torch.sum(self.activations, dim=1).squeeze(0)
        heatmap = torch.clamp(heatmap, min=0) # ReLU
        
        # Normalize between 0 and 1
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap.cpu().numpy(), class_idx
        
    def release(self):
        # Remove hooks to prevent memory leaks
        self.forward_hook.remove()
        self.backward_hook.remove()

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    
    # Transform matching training setup
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor = transform(img).unsqueeze(0) # add batch dim
    return tensor, original_size

def save_and_display_gradcam(img_path, heatmap, save_path, original_size, alpha=0.4):
    # Load original image in OpenCV
    img = cv2.imread(img_path)
    img = cv2.resize(img, original_size)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, original_size)
    
    # Scale heatmap to [0, 255] and convert to colormap
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    # Superimpose heatmap and original image
    superimposed_img = heatmap_colored * alpha + img
    
    # Normalize to [0, 255] range and save
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, superimposed_img)

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load class names
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    num_classes = len(classes)

    # Initialize model architecture
    print("Loading model...")
    model = models.inception_v3(pretrained=False, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load trained weights
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"WARNING: Model weights not found at {args.model_path}. Running with random initialization.")
        
    model = model.to(device)
    
    # Identify the target layer for hooks (last convolutional layer of Inception V3)
    # In InceptionV3, the last conv block is 'Conv2d_7b_1x1'
    target_layer = model.Conv2d_7b_1x1
    
    # Preprocess image
    x, original_size = preprocess_image(args.image_path)
    x = x.to(device)
    
    # Run Grad-CAM
    cam_generator = GradCAM(model, target_layer)
    heatmap, predicted_class_idx = cam_generator(x)
    cam_generator.release()
    
    predicted_class_name = classes[predicted_class_idx]
    print(f"Predicted Class: {predicted_class_name} (Index: {predicted_class_idx})")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    image_name = os.path.basename(args.image_path)
    save_path = os.path.join(args.output_dir, f"gradcam_{image_name}")
    
    # Save superimposed output
    save_and_display_gradcam(args.image_path, heatmap, save_path, original_size, alpha=0.4)
    print(f"Grad-CAM heatmap overlay saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to target image')
    parser.add_argument('classes', type=str, help='Path to classes txt file')
    parser.add_argument('model_path', type=str, help='Path to trained model file (.pt)')
    parser.add_argument('output_dir', type=str, help='Directory where result images will be saved')
    
    args = parser.parse_args()
    main(args)
