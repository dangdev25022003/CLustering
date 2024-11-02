import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image
import torch
from model_lib.embedding import Embedding

# Đường dẫn tới ảnh cần tìm kiếm
image_path = "data_embedding_t4/cluster_18/1712275702023.jpg"

# Đường dẫn tới file lưu embeddings và thông tin các ảnh
embeddings_path = 'embeddings/embedding_t4_04.npy'

# Thư mục chứa các ảnh
image_directory = "/media/dangph/data-1tb1/data_donghonuoc/2024/04_img"

# Thư mục để lưu các ảnh tương tự nhất
save_folder = "results/t4_01_lp"

# Bước 1: Tạo embedding từ ảnh cần tìm kiếm
def get_image_embedding(image_path):
    embedding = Embedding.getInstance()

    # Đọc ảnh từ file
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image {image_path} not found or failed to read.")
        return None

    # # Chuyển đổi từ BGR sang RGB (nếu cần)
    # img = img[:, :, ::-1]
    
    # Tạo embedding vector cho ảnh
    vec = embedding([img], ["image_to_search"])
    
    # Lấy embedding vector từ kết quả
    return vec[0]['vector']

# Bước 2: Tải danh sách các embeddings đã lưu
def load_saved_embeddings(embeddings_path, image_directory):
    # Tải file embeddings, cho phép pickle
    list_imgs = [f for f in os.listdir(image_directory) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
    all_embeddings = np.load(embeddings_path, allow_pickle=True)
    print(all_embeddings.shape)
    # Giả sử mỗi phần tử trong `all_embeddings` là một dict, ta cần lấy 'vector' từ mỗi phần tử
    all_vectors = np.array([embedding['vector'] for embedding in all_embeddings])
    allVectors = {}
    for i, image in tqdm(enumerate(list_imgs), total=len(list_imgs)):
        # Get the feature vector representation of the image
        vec = all_embeddings[i]
        # Store the feature vector in the allVectors dictionary, with the image filename as the key
        allVectors[image] = vec
    # Tải thông tin về các ảnh đã nhúng từ file JSON

    # Lấy danh sách tên ảnh
    image_names = list(allVectors.keys())
    
    return all_vectors, image_names

# Bước 3: Tính độ tương đồng và tìm kiếm các ảnh gần nhất
def search_similar_images(search_embedding, all_embeddings, image_names, image_directory, top_n=500, save_folder=None):
    # Tính cosine similarity giữa ảnh cần tìm và các ảnh đã lưu
    similarity_scores = cosine_similarity([search_embedding], all_embeddings)[0]
    
    # Sắp xếp theo độ tương đồng giảm dần
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    # Đảm bảo thư mục lưu ảnh tồn tại
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    # In ra top N kết quả giống nhau nhất và lưu ảnh
    print(f"Top {top_n} ảnh có độ tương đồng cao nhất:")
    for i in tqdm(range(top_n)):
        idx = sorted_indices[i]
        image_name = image_names[idx]
        similarity_score = similarity_scores[idx]

        # print(f"Ảnh: {image_name}, Similarity: {similarity_score:.4f}")

        # Tạo đường dẫn đầy đủ tới ảnh
        image_file_path = os.path.join(image_directory, image_name)

        # Đọc ảnh từ file và lưu vào thư mục đích
        if os.path.exists(image_file_path):
            image_to_save = cv2.imread(image_file_path)
            if image_to_save is not None:
                # Lưu ảnh vào thư mục kết quả
                save_path = os.path.join(save_folder, f'{similarity_score:.4f}.jpg')
                cv2.imwrite(save_path, image_to_save)
                # print(f"Đã lưu ảnh: {save_path}")
            else:
                print(f"Không thể đọc ảnh: {image_name}")
        else:
            print(f"Không tìm thấy ảnh: {image_file_path}")

# Bước 4: Chạy các bước trên để tìm ảnh giống nhau
def main(image_path, embeddings_path, image_directory, save_folder):
    # Tạo embedding cho ảnh cần tìm
    search_embedding = get_image_embedding(image_path)
    if search_embedding is None:
        return

    # Tải embeddings và tên ảnh đã lưu
    all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
    
    # Tìm và in ra các ảnh có độ tương đồng cao nhất và lưu ảnh
    search_similar_images(search_embedding, all_embeddings, image_names, image_directory, top_n=100, save_folder=save_folder)

# Chạy chương trình
if __name__ == "__main__":
    main(image_path, embeddings_path, image_directory, save_folder)
