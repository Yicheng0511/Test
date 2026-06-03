import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.utils.data
import pandas as pd
import os
import random
from tqdm import tqdm
import albumentations as A
import gc

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="0"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"]="1"

if torch.cuda.is_available():
    try:
        from torch.amp import autocast, GradScaler
    except ImportError:
        from torch.cuda.amp import autocast, GradScaler

    torch.backends.cudnn.benchmark = True
    scaler = GradScaler()
    torch.cuda.empty_cache()


MODEL_PATH = r"E:\Lungs_Segment_Info_Folder\unet_model.pth"
ANNOTATION_PATH = r"E:\Lungs_Segment_Info_Folder\annotations.csv"
IMG_DIR = r"E:\Lungs_CT_Dataset"

train_loss_list = []
val_loss_list = []
train_list = []
val_list = []

ann = pd.read_csv(ANNOTATION_PATH)
mhd_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".mhd")]

train_transform = A.Compose([
    A.Affine(translate_percent=0.08, rotate=(-10, 10), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=20, sigma=5, p=0.2),
    A.GaussNoise(std_range=(0.001, 0.005), p=0.3)
], additional_targets={"mask": "mask"})


class SegViTBottleneck(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(512, embed_dim, 4, 4)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, 2048, 0.1, batch_first=True),
            num_layers
        )
        self.up = nn.Upsample(scale_factor=4, mode='bilinear') # 4→16

    def forward(self, x):
        batch, channel, height, width = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1,2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.transpose(1,2).reshape(batch, channel, height//4, width//4)
        x = self.up(x)
        return x


class Unet3plus(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.e1 = self.conv_block(1, 64)
        self.e2 = self.conv_block(64, 128)
        self.e3 = self.conv_block(128, 256)
        self.e4 = self.conv_block(256, 512)

        self.bottleneck = SegViTBottleneck()

        self.d4 = self.conv_block(320, 64)
        self.d3 = self.conv_block(320, 64)
        self.d2 = self.conv_block(320, 64)
        self.d1 = self.conv_block(320, 64)

        self.conv1 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv2 = nn.Conv2d(128, 64, 3, padding="same")
        self.conv3 = nn.Conv2d(256, 64, 3, padding="same")
        self.conv4 = nn.Conv2d(512, 64, 3, padding="same")
        self.conv5 = nn.Conv2d(512, 64, 3, padding="same")
        self.conv_dec = nn.Conv2d(64, 64, 3, padding="same")

        self.pool = nn.MaxPool2d(2, 2)

        self.out1 = nn.Conv2d(64, 1, 1)
        self.out2 = nn.Conv2d(64, 1, 1)
        self.out3 = nn.Conv2d(64, 1, 1)
        self.out4 = nn.Conv2d(64, 1, 1)
        self.out5 = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.bottleneck(e4)

        e1_d4 = self.pool(self.pool(self.pool(e1)))
        e2_d4 = self.pool(self.pool(e2))
        e3_d4 = self.pool(e3)
        e5_d4 = F.interpolate(e5, e4.shape[2:], mode="bilinear", align_corners=False)

        e1_d4 = self.conv1(e1_d4)
        e2_d4 = self.conv2(e2_d4)
        e3_d4 = self.conv3(e3_d4)
        e4_d4 = self.conv4(e4)
        e5_d4 = self.conv5(e5_d4)
        d4 = self.d4(torch.cat([e5_d4, e4_d4, e3_d4, e2_d4, e1_d4], 1))

        e1_d3 = self.pool(self.pool(e1))
        e2_d3 = self.pool(e2)
        d4_d3 = F.interpolate(d4, e3.shape[2:], mode="bilinear", align_corners=False)
        e5_d3 = F.interpolate(e5, e3.shape[2:], mode="bilinear", align_corners=False)

        e1_d3 = self.conv1(e1_d3)
        e2_d3 = self.conv2(e2_d3)
        e3_d3 = self.conv3(e3)
        d4_d3 = self.conv_dec(d4_d3)
        e5_d3 = self.conv5(e5_d3)
        d3 = self.d3(torch.cat([e5_d3, d4_d3, e3_d3, e2_d3, e1_d3], 1))

        e1_d2 = self.pool(e1)
        d3_d2 = F.interpolate(d3, e2.shape[2:], mode="bilinear", align_corners=False)
        d4_d2 = F.interpolate(d4, e2.shape[2:], mode="bilinear", align_corners=False)
        e5_d2 = F.interpolate(e5, e2.shape[2:], mode="bilinear", align_corners=False)

        e1_d2 = self.conv1(e1_d2)
        e2_d2 = self.conv2(e2)
        d3_d2 = self.conv_dec(d3_d2)
        d4_d2 = self.conv_dec(d4_d2)
        e5_d2 = self.conv5(e5_d2)
        d2 = self.d2(torch.cat([d3_d2, d4_d2, e5_d2, e2_d2, e1_d2], 1))

        d2_d1 = F.interpolate(d2, e1.shape[2:], mode="bilinear", align_corners=False)
        d3_d1 = F.interpolate(d3, e1.shape[2:], mode="bilinear", align_corners=False)
        d4_d1 = F.interpolate(d4, e1.shape[2:], mode="bilinear", align_corners=False)
        e5_d1 = F.interpolate(e5, e1.shape[2:], mode="bilinear", align_corners=False)

        e1_d1 = self.conv1(e1)
        d2_d1 = self.conv_dec(d2_d1)
        d3_d1 = self.conv_dec(d3_d1)
        d4_d1 = self.conv_dec(d4_d1)
        e5_d1 = self.conv5(e5_d1)
        d1 = self.d1(torch.cat([d2_d1, d3_d1, d4_d1, e5_d1, e1_d1], 1))

        sup1 = self.out1(d1)
        sup2 = self.out2(d2)
        sup3 = self.out3(d3)
        sup4 = self.out4(d4)
        sup5 = self.out5(e5)

        sup1 = F.interpolate(sup1, x.shape[2:], mode='bilinear', align_corners=False)
        sup2 = F.interpolate(sup2, x.shape[2:], mode='bilinear', align_corners=False)
        sup3 = F.interpolate(sup3, x.shape[2:], mode='bilinear', align_corners=False)
        sup4 = F.interpolate(sup4, x.shape[2:], mode='bilinear', align_corners=False)
        sup5 = F.interpolate(sup5, x.shape[2:], mode='bilinear', align_corners=False)

        return sup1, sup2, sup3, sup4, sup5

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same", bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding="same", bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True), 
            nn.Dropout2d(0.3)
        )
    

class LungsDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, all_mhd_files, transform=None):
        self.cache = {}
        self.all_mhd = all_mhd_files
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        info = self.file_list[idx]
        if "is_negative" in info and info["is_negative"]:
            file_name = np.random.choice(self.all_mhd)
            if file_name not in self.cache:
                self.cache[file_name] = self.load_mhd_with_info(os.path.join(IMG_DIR, file_name))
                if len(self.cache) % 15 == 0:
                    gc.collect()
            ct, *_ = self.cache[file_name]
            z = np.random.randint(0, ct.shape[0])
            img = ct[z]
            if img.mean() < 10:  
                return self.__getitem__(random.randint(0, len(self)-1))
            mask = np.zeros_like(img, np.float32)

        else:
            file_name = info["file_name"]
            x_pos = info["x_pos"]
            y_pos = info["y_pos"]
            z_pos = info["z_pos"]
            diameter = info["diameter"]
            radius_mm = max(diameter / 2, 2)

            if file_name not in self.cache:
                self.cache[file_name] = self.load_mhd_with_info(os.path.join(IMG_DIR, file_name))
                if len(self.cache) % 15 == 0:
                    gc.collect()
            ct, origin, spacing, direction = self.cache[file_name]

            x, y, z = self.world_to_pixel(x_pos, y_pos, z_pos, origin, spacing, direction)
            
            if 0 <= z < ct.shape[0]:
                img = ct[z]
                mask = self.create_2d_mask(img.shape, x, y, radius_mm, spacing[0], spacing[1])
            else:
                img = np.zeros((512,512), np.float32)
                mask = np.zeros((512,512), np.float32)

        img = self.normalize_slice(img)
        img = np.asarray(img, dtype=np.float32)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

    @staticmethod
    def load_mhd_with_info(path):
        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)   # (z, y, x)
        origin = np.array(image.GetOrigin())    # (x, y, z)
        spacing = np.array(image.GetSpacing())  # (x, y, z)
        direction = np.array(image.GetDirection())
        del image
        return array, origin, spacing, direction
    
    @staticmethod
    def normalize_slice(slice_2d):
        min_hu = -1000
        max_hu = 400
        slice_2d = np.clip(slice_2d, min_hu, max_hu)
        slice_2d = (slice_2d - min_hu) / (max_hu - min_hu)
        return slice_2d
    
    @staticmethod
    def world_to_pixel(x_pos, y_pos, z_pos, origin, spacing, direction):
        dx = direction[0]
        dy = direction[4]
        dz = direction[8]

        x_index = int(round((x_pos - origin[0]) / spacing[0] * dx))
        y_index = int(round((y_pos - origin[1]) / spacing[1] * dy))
        z_index = int(round((z_pos - origin[2]) / spacing[2] * dz))
        return x_index, y_index, z_index
    
    @staticmethod
    def create_2d_mask(shape, x_index, y_index, radius_mm, x_spacing, y_spacing):
        mask = np.zeros(shape, dtype=np.float32)

        yy, xx = np.ogrid[:shape[0], :shape[1]]
        dx = (xx - x_index) * x_spacing
        dy = (yy - y_index) * y_spacing
        dist = np.sqrt(dx**2 + dy**2)

        mask[dist <= radius_mm] = 1
        return mask


class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=4/3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs, masks):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.reshape(-1)
        masks = masks.reshape(-1)

        TP = torch.sum(outputs * masks)
        FP = torch.sum(outputs * (1 - masks))
        FN = torch.sum((1 - outputs) * masks)

        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        loss = torch.pow(1 - tversky, self.gamma)
        return loss
    

class EarlyStopping:
    def __init__(self, patience=7, threshold=0.0001):
        self.patience = patience
        self.threshold = threshold
        self.best_value = -float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, value):
        if value > self.best_value + self.threshold:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def tversky_coefficient(outputs, masks, alpha=0.3, beta=0.7, smooth=1e-6):
    outputs = torch.sigmoid(outputs)
    outputs = outputs.reshape(-1)
    masks = masks.reshape(-1)
    
    TP = torch.sum(outputs * masks)
    FP = torch.sum(outputs * (1 - masks))
    FN = torch.sum((1 - outputs) * masks)
    
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return tversky


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):

    total_start_time = time.time()
    early_stopping = EarlyStopping(8)
    best_val_tversky = 0.0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        val_loss = 0.0
        val_tversky = 0.0
        running_loss = 0.0
        running_tversky = 0.0

        model.train()
        for images, masks in tqdm(train_loader, "Train"):
            if torch.any(torch.isnan(images)) or torch.any(torch.isnan(masks)):
                continue
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                s1, s2, s3, s4, s5 = model(images)
                loss_s1 = criterion(s1, masks)
                loss_s2 = criterion(s2, masks)
                loss_s3 = criterion(s3, masks)
                loss_s4 = criterion(s4, masks)
                loss_s5 = criterion(s5, masks)

                total_loss = (loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5) / 5
                
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item() 
            running_tversky += tversky_coefficient(s1, masks).item()

            del images, masks, s1, s2, s3, s4, s5, total_loss
            gc.collect()

        epoch_loss = running_loss / len(train_loader)
        epoch_tversky = running_tversky / len(train_loader)

        model.eval()

        with torch.no_grad:
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                s1, s2, s3, s4, s5 = model(images)
                loss_s1 = criterion(s1, masks)
                loss_s2 = criterion(s2, masks)
                loss_s3 = criterion(s3, masks)
                loss_s4 = criterion(s4, masks)
                loss_s5 = criterion(s5, masks)

                total_loss = (loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5) / 5

                val_loss += total_loss.item()
                val_tversky += tversky_coefficient(s1, masks).item()

        val_loss /= len(val_loader)
        val_tversky /= len(val_loader)

        train_loss_list.append(epoch_loss)
        val_loss_list.append(val_loss)
        train_list.append(epoch_tversky)
        val_list.append(val_tversky)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        gc.collect()
        torch.cuda.empty_cache()

        print(f'Epoch {epoch+1}/{num_epochs}, Duration: {epoch_duration:.2f} seconds, '
            f'Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, '
            f'Tversky Coefficient: {epoch_tversky:.4f}, Validation Tversky: {val_tversky:.4f}')     
           
        if val_tversky > best_val_tversky:
            best_val_tversky = val_tversky
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model saved (Val Tversky: {best_val_tversky:.4f})")
        
        if early_stopping(val_tversky):
            print("Early Stopping Activated! Training Stopped")
            break


    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f'Total training time: {total_duration:.2f} seconds')


def predict(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        sup1, *_ = model(image_tensor)
        pred = torch.sigmoid(sup1)
    return pred


def visualize_prediction(image_tensor, prediction, mask_tensor):
    image = image_tensor.squeeze().cpu().numpy()
    pred = prediction.squeeze().cpu().numpy()
    mask = mask_tensor.squeeze().cpu().numpy()

    pred_bin = (pred > 0.4).astype(np.uint8)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title('Predicted Mask')
    plt.imshow(image, cmap='gray')
    plt.imshow(pred_bin, cmap='Reds', alpha=0.7)

    plt.subplot(1, 3, 3)
    plt.title('Ground Truth Mask')
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='Reds', alpha=0.7)

    plt.show()



def plot_loss_lr(train_loss, val_loss, train, val):
    epochs_range = range(1, len(train_loss)+1)
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(epochs_range, train_loss, label='Train Loss', color='#2E86AB', linewidth=2)
    ax1.plot(epochs_range, val_loss, label='Val Loss', color='#A23B72', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    ax1.set_title('Train & Validation Loss Curve')

    ax2.plot(epochs_range, train, label='Train Tversky', color='#2E86AB', linewidth=2)
    ax2.plot(epochs_range, val, label='Val Tversky', color='#A23B72', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Tversky')
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    ax2.set_title('Train & Validation tversky Curve')
    
    plt.tight_layout()
    plt.show()


def add_negative_samples(positive_list, all_mhd_files, negative_ratio):
    pos_len = len(positive_list)
    neg_len = int(pos_len * negative_ratio)
    
    neg_samples = []
    for _ in range(neg_len):
        file = np.random.choice(all_mhd_files)
        neg_samples.append({
            "file_name": file,
            "is_negative": True
        })
    return positive_list + neg_samples


if __name__ == "__main__":
    model = Unet3plus()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = TverskyFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 6e-5, weight_decay=1e-5)

    positive_samples = []
    for f in mhd_files:
        sid = f.replace(".mhd","")
        sub = ann[ann["seriesuid"]==sid]
        for _, row in sub.iterrows():
            positive_samples.append({
                "file_name":f,
                "x_pos":row.coordX,
                "y_pos":row.coordY,
                "z_pos":row.coordZ,
                "diameter":row.diameter_mm
            })
    
    full = positive_samples
    np.random.shuffle(full)
    n = len(full)
    n_train = int(0.8*n)
    n_val = int(0.1*n)
    train_files = full[:n_train]
    val_files = full[n_train:n_train+n_val]
    test_files = full[n_train+n_val:]

    negative_ratio = 0.0
    train_files = add_negative_samples(train_files, mhd_files, negative_ratio)
    val_files = add_negative_samples(val_files, mhd_files, negative_ratio)
    test_files = add_negative_samples(test_files, mhd_files, negative_ratio)
    np.random.shuffle(test_files)
    train_dataset = LungsDataset(train_files, mhd_files, train_transform)
    val_dataset = LungsDataset(val_files, mhd_files)
    test_dataset = LungsDataset(test_files, mhd_files)

    train_loader = torch.utils.data.DataLoader(train_dataset, 3, True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 3, False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 3, False, drop_last=True)

    print("Total files:", n)
    print("Total Training Dataset:", len(train_dataset))
    print("Original positive sample:", len(full[:n_train]))

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, device, weights_only=True))
        train_model(model, criterion, optimizer, train_loader, val_loader, 150)
        print("Model loaded successfully from checkpoint.")
    else:
        train_model(model, criterion, optimizer, train_loader, val_loader, 150)

    model.load_state_dict(torch.load(MODEL_PATH, device, weights_only=True))
    plot_loss_lr(train_loss_list, val_loss_list, train_list, val_list)
    model.eval()
    print(len(test_dataset))

    for test_data in range(len(test_dataset)):
        test_image, test_mask_image = test_dataset[test_data]
        test_image = test_image.unsqueeze(0)
        test_mask_image =  test_mask_image.unsqueeze(0)

        prediction = predict(model, test_image)
        visualize_prediction(test_image, prediction, test_mask_image)

        plt.close("all")
        del test_image, test_mask_image, prediction
        gc.collect()
