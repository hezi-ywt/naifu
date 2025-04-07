import os
import shutil
from tqdm import tqdm
from PIL import Image
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("error", category=UserWarning, message=".*Corrupt EXIF data.*")

def filter_by_tag(img_pth, txt_pth=None, tag_list=["comic"]):
    if txt_pth is None:
        print("txt_pth is None")
        return False
    with open(txt_pth, "r") as f:
        txt = f.read()
        
    tag_ = txt.split(", ")
    for tag in tag_list:
        if tag in tag_:
            print(f"tag found: {tag}")
            return True

    return False

def filter_by_size_ratio(img_pth, ratio=4):
    img = Image.open(img_pth)
    size_ratio = img.size[0] / img.size[1]
    if size_ratio > ratio:
        print(f"size ratio exceed: {img_pth}")
        return True
    elif size_ratio < 1/ratio:
        print(f"size ratio exceed: {img_pth}")
        return True
    else:
        return False
                
def filter_by_size_limit(img_pth, size_limit=80000000):
    try:
        img = Image.open(img_pth)
        if img.size[0] * img.size[1] > size_limit:
            print(f"size limit exceed: {img_pth}")
            return True
        else:
            return False
    except Exception as e:
        print(f"Cannot read image file: {img_pth}, error message: {str(e)}, pass this file")
        return True

def image_filter(img_pth, txt_pth, tag_list, ratio, trash_dir):
    try:
        if filter_by_size_limit(img_pth):
            return True
        
        with Image.open(img_pth) as img:
            if img.mode == "RGBA":
                extrema = img.getextrema()
                if extrema[3][0] < 255:
                    return True
            elif img.mode == "P":
                transparent = img.info.get("transparency", -1)
                for _, index in img.getcolors():
                    if index == transparent:
                        return True

        if filter_by_size_ratio(img_pth, ratio):
            return True
        
        return False
    except Exception as e:
        print(f"Error processing file: {img_pth} - {e}")
        if "Corrupt EXIF data" in str(e):
            print(f"corrupt EXIF data: {img_pth}")
            move_to_trash(img_pth, trash_dir)
            txt_pth = os.path.splitext(img_pth)[0] + ".txt"
            if os.path.exists(txt_pth):
                move_to_trash(txt_pth, trash_dir)
            print(f"Moved {img_pth} and corresponding .txt file to trash due to corrupt EXIF.")
        return True

def move_to_trash(pth, trash_dir):
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)
    shutil.move(pth, os.path.join(trash_dir, os.path.basename(pth)))

def judge_EXIT(image_pth):
    warnings.filterwarnings("error")
    try:
        with Image.open(image_pth) as image:
            if has_transparency(image):
                print(f"has transparency")
                return True
            else:
                return False
    except Exception as e:
        if "Corrupt EXIF data" in str(e):
            print(f"corrupt EXIF data: {image_pth}")
        else:
            print(f"Error processing image: {image_pth} - {e}")
        return True

def has_transparency(img):
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True
    return False

IMG_ENDWITH = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
TAG_LIST = ["comic", "manga", "4koma", "scan"]

def filter_and_move(img_pth, trash_dir="/mnt/data/trash", ratio=4):
    txt_pth = os.path.splitext(img_pth)[0] + ".txt"
    if image_filter(img_pth, txt_pth, TAG_LIST, ratio=ratio, trash_dir=trash_dir):
        print(f"move to trash: {os.path.basename(img_pth)}")
        if os.path.exists(txt_pth):
            move_to_trash(img_pth, trash_dir)
            move_to_trash(txt_pth, trash_dir)
        else:
            move_to_trash(img_pth, trash_dir)
            # print(f"txt file not found for post_id: {os.path.basename(img_pth)}")

def filter_dir(img_dir):
    files = []
    # 递归遍历所有子文件夹
    for root, _, filenames in os.walk(img_dir):
        for filename in filenames:
            if any(filename.endswith(end) for end in IMG_ENDWITH):
                files.append(os.path.join(root, filename))
    
    print(f"找到 {len(files)} 个图像文件")
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(filter_and_move, files), total=len(files), desc='Processing'):
            pass

def main():
    img_dir = "/mnt/public/dataset/danbooru2024-webp-4Mpixel/images_untar"
    out_dir = "/mnt/public/dataset/danbooru2024-webp-4Mpixel/images_untar_trash"
    ratio = 4
    filter_dir(img_dir)

if __name__ == "__main__":
    main()
