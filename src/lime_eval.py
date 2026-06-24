import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from lime import lime_image

# Preprocessing transforms for PyTorch InceptionV3
preprocess_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def py_predict_wrapper(model, device):
    """
    Creates a prediction function compatible with LIME,
    which expects a numpy array of shape (N, H, W, C) and returns probability array (N, num_classes)
    """
    def predict_fn(images):
        model.eval()
        tensors = []
        for img in images:
            # LIME images are numpy arrays in range [0, 1] or [0, 255]
            # Convert to PIL Image first to apply standard transform
            if img.max() <= 1.0:
                img = img * 255.0
            pil_img = Image.fromarray(np.uint8(img))
            tensor = preprocess_transform(pil_img)
            tensors.append(tensor)
            
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(x)
            # Inception V3 outputs logits. We apply Softmax to get probabilities.
            probs = torch.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()
        
    return predict_fn

def binary_mask_from_gt(gt_path):
    """
    Loads ground truth mask image and converts it to binary mask (299x299).
    Following your method: pixels that are not pure white (value < 1.0 or < 255)
    are considered part of the object (True Positive pixels).
    """
    gt_img = Image.open(gt_path).convert('L').resize((299, 299))
    gt_array = np.array(gt_img) / 255.0
    
    # Object pixels are non-white (black/grey)
    # Binary mask: 1 for object, 0 for background
    binary_mask = (gt_array < 0.95).astype(np.uint8)
    return binary_mask

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class list
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    num_classes = len(classes)

    # Initialize model
    model = models.inception_v3(pretrained=False, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Model weights not found at: {args.model_path}")
        
    model = model.to(device)
    
    # Initialize LIME Explainer
    explainer = lime_image.LimeImageExplainer()
    predict_fn = py_predict_wrapper(model, device)
    
    # Output structure
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    
    # Loop through target images
    img_pattern = os.path.join(args.image_dir, "*.*")
    image_paths = [f for f in glob.glob(img_pattern) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_paths)} images for LIME evaluation.")

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        filename_only, ext = os.path.splitext(filename)
        print(f"[{idx+1}/{len(image_paths)}] Explaining image: {filename}")
        
        # Load and predict
        pil_img = Image.open(img_path).convert('RGB').resize((299, 299))
        img_np = np.array(pil_img) / 255.0 # normalize to [0, 1] for LIME
        
        # Check prediction
        probs = predict_fn([img_np])[0]
        pred_class_idx = np.argmax(probs)
        pred_class_name = classes[pred_class_idx]
        pred_prob = probs[pred_class_idx] * 100.0
        
        print(f"  Predicted: {pred_class_name} ({pred_prob:.2f}%)")
        
        # Look for Ground Truth mask
        gt_path = os.path.join(args.gt_mask_dir, filename)
        has_gt = os.path.exists(gt_path)
        
        if not has_gt:
            print(f"  WARNING: Ground Truth mask not found at: {gt_path}. Skipping metrics for this file.")
            # Still record classification prediction
            results.append({
                'Filename': filename,
                'Predicted_Class': pred_class_name,
                'Probability_%': pred_prob,
                'Accuracy_%': 0, 'Precision_%': 0, 'Recall_%': 0, 'Specificity_%': 0, 'F1_Score_%': 0
            })
            continue

        # Get LIME explanation
        # hide_color=0 replaces superpixels with black when turned off
        explanation = explainer.explain_instance(
            img_np, 
            predict_fn, 
            top_labels=1, 
            hide_color=0, 
            num_samples=args.num_samples
        )
        
        # Get explanation image and binary mask (top features)
        # mask is 2D binary array: 1 where features are present, 0 elsewhere
        temp, lime_mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=args.num_features, 
            hide_rest=True
        )
        
        # Load ground truth mask (binary)
        gt_mask = binary_mask_from_gt(gt_path)
        
        # Calculate confusion matrix on pixel levels
        # TP: LIME active and GT active
        mask_tp = gt_mask * lime_mask
        # TN: LIME inactive and GT inactive
        mask_tn = (1 - gt_mask) * (1 - lime_mask)
        # FP: LIME active and GT inactive
        mask_fp = (1 - gt_mask) * lime_mask
        # FN: LIME inactive and GT active
        mask_fn = gt_mask * (1 - lime_mask)
        
        # Count pixel rates (%)
        total_pixels = 299 * 299
        tp = np.count_nonzero(mask_tp) / total_pixels * 100.0
        tn = np.count_nonzero(mask_tn) / total_pixels * 100.0
        fp = np.count_nonzero(mask_fp) / total_pixels * 100.0
        fn = np.count_nonzero(mask_fn) / total_pixels * 100.0
        
        # Compute overlapping statistics
        acc = (tp + tn) / (tp + tn + fp + fn) * 100.0
        pre = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0
        spe = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0
        f1 = (2 * (pre * rec) / (pre + rec)) if (pre + rec) > 0 else 0
        
        print(f"  Overlap Metrics -> Accuracy: {acc:.2f}%, Precision: {pre:.2f}%, Recall/Recall: {rec:.2f}%, F1: {f1:.2f}%")
        
        # Save LIME visualization
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(mark_boundaries(img_np, lime_mask))
        ax[0].set_title(f"LIME Features ({pred_class_name})")
        ax[0].axis('off')
        
        ax[1].imshow(gt_mask, cmap='gray')
        ax[1].set_title("Ground Truth Mask")
        ax[1].axis('off')
        
        vis_save_path = os.path.join(args.output_dir, f"lime_vis_{filename_only}.png")
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()
        
        results.append({
            'Filename': filename,
            'Predicted_Class': pred_class_name,
            'Probability_%': pred_prob,
            'Accuracy_%': acc,
            'Precision_%': pre,
            'Recall_%': rec,
            'Specificity_%': spe,
            'F1_Score_%': f1
        })
        
    # Write summary spreadsheet
    df = pd.DataFrame(results)
    excel_path = os.path.join(args.output_dir, "LIME_quantitative_evaluation.xlsx")
    
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='LIME_Evaluation', index=False)
    
    # Apply styling
    workbook = writer.book
    worksheet = writer.sheets['LIME_Evaluation']
    header_format = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1
    })
    
    # Write header explicitly with styling
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
        
    writer.close()
    print(f"\nQuantitative evaluation complete! Results saved to: {excel_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='Directory of validation images')
    parser.add_argument('gt_mask_dir', type=str, help='Directory of ground truth masks')
    parser.add_argument('classes', type=str, help='Path to classes txt file')
    parser.add_argument('model_path', type=str, help='Path to trained PyTorch model (.pt)')
    parser.add_argument('output_dir', type=str, help='Directory to save visualization and spreadsheet')
    parser.add_argument('--num_samples', type=int, default=1000, help='LIME perturbation samples')
    parser.add_argument('--num_features', type=int, default=10, help='LIME top features to evaluate')
    
    args = parser.parse_args()
    main(args)
