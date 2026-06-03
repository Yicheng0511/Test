import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
from tqdm import tqdm

# ===================== 路径 =====================
ANNOTATION_PATH = r"E:\Lungs_Segment_Info_Folder\annotations.csv"
IMG_DIR = r"E:\Lungs_CT_Dataset"
SAVE_DIR = r"E:\Lungs_POS_NPY"  # 只存正样本
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 工具函数 =====================
def load_mhd(path):
    itk = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(itk)
    origin = np.array(itk.GetOrigin())
    spacing = np.array(itk.GetSpacing())
    direction = np.array(itk.GetDirection())
    del itk
    return arr, origin, spacing, direction

def normalize(img):
    min_hu = -1000
    max_hu = 400
    img = np.clip(img, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)
    return img.astype(np.float32)

def world_to_pixel(xp, yp, zp, origin, spacing, dirs):
    dx, dy, dz = dirs[0], dirs[4], dirs[8]
    xi = int(round((xp - origin[0])/spacing[0] * dx))
    yi = int(round((yp - origin[1])/spacing[1] * dy))
    zi = int(round((zp - origin[2])/spacing[2] * dz))
    return xi, yi, zi

def create_mask_2d(shape, x, y, r, sx, sy):
    m = np.zeros(shape, np.float32)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt(((xx-x)*sx)**2 + ((yy-y)*sy)**2)
    m[dist <= r] = 1
    return m

# ===================== 只预处理 正样本 =====================
df = pd.read_csv(ANNOTATION_PATH)
mhd_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".mhd")]

positive_list = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="预处理正样本"):
    seriesuid = row["seriesuid"]
    fname = f"{seriesuid}.mhd"
    path = os.path.join(IMG_DIR, fname)
    
    if not os.path.exists(path):
        continue

    # 读取CT
    ct, ori, spa, direction = load_mhd(path)

    # 坐标转换
    x, y, z = world_to_pixel(row.coordX, row.coordY, row.coordZ, ori, spa, direction)

    if not (0 <= z < ct.shape[0]):
        continue
    
    img = ct[z]

    # 生成 2D mask
    r = max(row.diameter_mm/2, 2)
    mask_slice = create_mask_2d(img.shape, x, y, r, spa[0], spa[1])
    img = normalize(img)

    # 保存：只存结节所在层
    save_name = f"pos_{seriesuid}_{idx}.npz"
    save_path = os.path.join(SAVE_DIR, save_name)
    np.savez_compressed(save_path, img=img, mask=mask_slice)
    
    positive_list.append({
        "file_path": save_path,
        "is_negative": False
    })

# 保存正样本列表
np.save(os.path.join(SAVE_DIR, "positive_list.npy"), positive_list)
print(f"✅ Proprocessed frinished! Total: {len(positive_list)} slices")
