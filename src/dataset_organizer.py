import os
import argparse
import shutil
import random

def split_data(source_images_dir, source_masks_dir, dest_root, class_name, split_ratio):
    # Create destination directories
    train_images_dir = os.path.join(dest_root, "images", "train", class_name)
    val_images_dir = os.path.join(dest_root, "images", "val", class_name)
    train_masks_dir = os.path.join(dest_root, "masks", "train", class_name)
    val_masks_dir = os.path.join(dest_root, "masks", "val", class_name)
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    
    has_masks = source_masks_dir is not None and os.path.exists(source_masks_dir)
    if has_masks:
        os.makedirs(train_masks_dir, exist_ok=True)
        os.makedirs(val_masks_dir, exist_ok=True)
        
    # Get all image files
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    all_files = os.listdir(source_images_dir)
    images = [f for f in all_files if os.path.isfile(os.path.join(source_images_dir, f)) and f.lower().endswith(valid_exts)]
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Copy files
    copied_images = 0
    copied_masks = 0
    
    # Process training set
    for img_name in train_images:
        # Copy image
        shutil.copy2(os.path.join(source_images_dir, img_name), os.path.join(train_images_dir, img_name))
        copied_images += 1
        
        # Copy mask if exists
        if has_masks:
            # The mask might have a different extension (.png vs .jpg) or match the name exactly
            base_name, _ = os.path.splitext(img_name)
            mask_png = base_name + ".png"
            mask_jpg = base_name + ".jpg"
            mask_jpeg = base_name + ".jpeg"
            
            found_mask = None
            for m_candidate in [mask_png, mask_jpg, mask_jpeg, img_name]:
                if os.path.exists(os.path.join(source_masks_dir, m_candidate)):
                    found_mask = m_candidate
                    break
            
            if found_mask:
                shutil.copy2(os.path.join(source_masks_dir, found_mask), os.path.join(train_masks_dir, found_mask))
                copied_masks += 1
                
    # Process validation set
    for img_name in val_images:
        # Copy image
        shutil.copy2(os.path.join(source_images_dir, img_name), os.path.join(val_images_dir, img_name))
        copied_images += 1
        
        # Copy mask if exists
        if has_masks:
            base_name, _ = os.path.splitext(img_name)
            mask_png = base_name + ".png"
            mask_jpg = base_name + ".jpg"
            mask_jpeg = base_name + ".jpeg"
            
            found_mask = None
            for m_candidate in [mask_png, mask_jpg, mask_jpeg, img_name]:
                if os.path.exists(os.path.join(source_masks_dir, m_candidate)):
                    found_mask = m_candidate
                    break
            
            if found_mask:
                shutil.copy2(os.path.join(source_masks_dir, found_mask), os.path.join(val_masks_dir, found_mask))
                copied_masks += 1
                
    print(f"Class '{class_name}': Split {len(images)} images -> {len(train_images)} train, {len(val_images)} val. (Copied {copied_masks} masks)")

def main():
    parser = argparse.ArgumentParser(description="Organizador de Datasets de Imagens e Máscaras (Split Treino/Val)")
    parser.add_argument("--source_dir", type=str, required=True, help="Diretório contendo as imagens brutas ou subpastas de classes")
    parser.add_argument("--dest_dir", type=str, default="dataset", help="Diretório de destino para o dataset estruturado")
    parser.add_argument("--split", type=float, default=0.8, help="Fração de imagens para treino (ex: 0.8)")
    parser.add_argument("--classes", type=str, required=True, help="Caminho para o arquivo classes.txt com os nomes das classes")
    args = parser.parse_args()
    
    # Load class names
    if not os.path.exists(args.classes):
        print(f"Erro: Arquivo de classes não encontrado em {args.classes}")
        return
        
    with open(args.classes, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Organizando dataset para as classes: {classes}")
    print(f"Destino: {args.dest_dir}")
    
    for class_name in classes:
        # Check for standard class subdirectory in source
        class_sub = os.path.join(args.source_dir, class_name)
        
        # Fallback names in the case of specific folders (e.g. dalmatian 2020_09_11_total)
        # We search the source directory for any folders matching the class name prefix
        source_img_path = None
        source_mask_path = None
        
        if os.path.exists(class_sub):
            source_img_path = class_sub
        else:
            # Look for subdirs like 'dalmatian 2020_09_11_total' or 'dalmatian' in source
            for d in os.listdir(args.source_dir):
                d_path = os.path.join(args.source_dir, d)
                if os.path.isdir(d_path) and class_name.lower() in d.lower():
                    source_img_path = d_path
                    break
                    
        if source_img_path is None:
            print(f"Aviso: Diretório para a classe '{class_name}' não encontrado no source. Pulando...")
            continue
            
        # Check for masks subdirectory
        # Masks can be inside the class folder itself (under 'mask') or in a parallel path
        mask_sub = os.path.join(source_img_path, "mask")
        if os.path.exists(mask_sub):
            source_mask_path = mask_sub
        else:
            # Check parallel folder under source_dir (like 3dogs/mask)
            pass
            
        split_data(source_img_path, source_mask_path, args.dest_dir, class_name, args.split)
        
    print("\nDataset organizado com sucesso!")
    print(f"Imagens de treino em: {os.path.join(args.dest_dir, 'images', 'train')}")
    print(f"Imagens de validação em: {os.path.join(args.dest_dir, 'images', 'val')}")

if __name__ == "__main__":
    main()
