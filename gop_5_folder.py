import os
import shutil
from tqdm import tqdm

def merge_folders(folder_list, destination_folder):
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Lấy danh sách các thư mục con có tên trùng nhau
    common_subfolders = set(os.listdir(folder_list[0]))  # Lấy các thư mục con của folder đầu tiên
    for folder in folder_list[1:]:
        subfolders = set(os.listdir(folder))
        common_subfolders = common_subfolders.intersection(subfolders)

    # Duyệt qua các thư mục trùng tên và gộp lại
    for subfolder in tqdm(common_subfolders, desc="Processing subfolders"):
        destination_subfolder = os.path.join(destination_folder, subfolder)
        if not os.path.exists(destination_subfolder):
            os.makedirs(destination_subfolder)

        # Gộp nội dung của folder nhỏ có tên trùng nhau
        for folder in folder_list:
            source_subfolder = os.path.join(folder, subfolder)
            items = os.listdir(source_subfolder)
            
            for item in tqdm(items, desc=f"Copying from {subfolder}", leave=False):
                source_item = os.path.join(source_subfolder, item)
                destination_item = os.path.join(destination_subfolder, item)

                # Nếu là file thì copy, nếu là folder thì copy cả thư mục
                if os.path.isfile(source_item):
                    shutil.copy2(source_item, destination_item)
                elif os.path.isdir(source_item):
                    if not os.path.exists(destination_item):
                        shutil.copytree(source_item, destination_item)
                    else:
                        # Gộp file nếu thư mục đã tồn tại
                        for sub_item in os.listdir(source_item):
                            shutil.copy2(os.path.join(source_item, sub_item), os.path.join(destination_item, sub_item))

# Danh sách các folder lớn
folders = [
    '/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_7/T4', 
    '/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_7/T5', 
    '/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_7/T5', 
    '/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_7/T7', 
    '/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_7/T8'
]

# Thư mục đích lưu các folder đã gộp
destination = '/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_7/T9'

merge_folders(folders, destination)

print(f"Tất cả các folder nhỏ đã được gộp vào {destination}")
