import os  
import cv2
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from model_lib.embedding import Embedding
from tqdm import tqdm
list_images = [
"/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_7/T4/1/1712481264831.jpg",
# '/home/dangph/Downloads/anh_xoay/1710048439054.jpg',
# '/home/dangph/Downloads/anh_xoay/1710048439054.jpg',
# '/home/dangph/Downloads/anh_xoay/1712909412108.jpg',
# '/home/dangph/Downloads/anh_xoay/1714094651577.jpg',
# '/home/dangph/Downloads/anh_xoay/1712132957938.jpg',
# '/home/dangph/Downloads/anh_xoay/1712043911921.jpg',
# '/home/dangph/Downloads/anh_xoay/1711593954998.jpg',
# '/home/dangph/Downloads/anh_xoay/1712487340408.jpg',
# '/home/dangph/Downloads/anh_xoay/1711931150271.jpg',
# '/home/dangph/Downloads/anh_xoay/1712191363335.jpg',
# '/home/dangph/Downloads/anh_xoay/1712571466017.jpg',
# '/home/dangph/Downloads/anh_xoay/1712458013831.jpg',
# '/home/dangph/Downloads/anh_xoay/1712538983247.jpg',
# '/home/dangph/Downloads/anh_xoay/1712539315898.jpg',
# '/home/dangph/Downloads/anh_xoay/1712968358360.jpg',
# '/home/dangph/Downloads/anh_xoay/1712180704324.jpg',
# '/home/dangph/Downloads/anh_xoay/1712910891638.jpg',
# '/home/dangph/Downloads/anh_xoay/1712279083609.jpg',
# '/home/dangph/Downloads/anh_xoay/1712183764674.jpg',
# '/home/dangph/Downloads/anh_xoay/1712185965568.jpg',
# '/home/dangph/Downloads/anh_xoay/1714090236699.jpg',
# '/home/dangph/Downloads/anh_xoay/1712974203090.jpg',
# '/home/dangph/Downloads/anh_xoay/1712226896790.jpg',
# '/home/dangph/Downloads/anh_xoay/1712136718597.jpg',
# '/mnt/sdb1/data_donghonuoc/2024/06_img/hinhanhdongho_1717982946645.jpg',
# '/home/dangph/Downloads/anh_xoay/1710056563286.jpg',
# '/home/dangph/Downloads/anh_xoay/1712224221806.jpg',
# '/home/dangph/Downloads/anh_xoay/1710151899180.jpg',
# '/home/dangph/Downloads/anh_xoay/1714095010608.jpg',
# '/home/dangph/Downloads/anh_xoay/1712217351659.jpg',
# '/home/dangph/Downloads/anh_xoay/1712367178045.jpg',
# '/home/dangph/Downloads/anh_xoay/1713055474027.jpg',
# '/home/dangph/Downloads/anh_xoay/1712660233660.jpg',
# '/home/dangph/Downloads/anh_xoay/1714092616869.jpg',
# '/home/dangph/Downloads/anh_xoay/1714110827415.jpg',
# '/home/dangph/Downloads/anh_xoay/1712974700772.jpg',
# '/home/dangph/Downloads/anh_xoay/1712910444205.jpg',
# '/home/dangph/Downloads/anh_xoay/1714088659937.jpg',
# '/home/dangph/Downloads/anh_xoay/1714096558014.jpg',
# '/home/dangph/Downloads/anh_xoay/1713175817198.jpg'"
]
count_nb = 1
for j in list_images:

    image_path = f"{j}"
# Đường dẫn tới ảnh cần tìm kiếm
# image_path = "/mnt/sdb1/data_donghonuoc/clustering/data_embedding_t4/cluster_0/1710895565715.jpg"

    # Đường dẫn tới file lưu embeddings và thông tin các ảnh
    embeddings_path = '/media/dangph/data-1tb1/data_donghonuoc/clustering/embeddings/embedding_full_SKU.npy'
    # json_path = '/mnt/sdb1/data_donghonuoc/clustering/embeddings_images_meters_t6.json'
    image_directory = "/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_7/T4/1"
    # Thư mục để lưu các ảnh tương tự nhất
    save_folder = f"/mnt/sdb1/data_donghonuoc/clustering/similar_images_results_4/{count_nb}"
    so_luong = 500
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

        # Thư mục nơi các ảnh đang lưu trữ (đảm bảo đường dẫn này là chính xác)
        # Giả sử JSON chứa tên ảnh tương đối hoặc tuyệt đối

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
            if os.path.exists(image_file_path):  # Kiểm tra xem ảnh có tồn tại không
                image_to_save = cv2.imread(image_file_path)  # Đọc ảnh gốc từ file
                if image_to_save is not None:
                    # Lưu ảnh với tên mới
                    save_path = os.path.join(save_folder, image_name)
                    cv2.imwrite(save_path, image_to_save)
                    # print(f"Đã lưu ảnh: {save_path}")
                else:
                    print(f"Không thể đọc ảnh: {image_name}")
            else:
                print(f"Không tìm thấy ảnh: {image_file_path}")

    # Bước 4: Chạy các bước trên để tìm ảnh giống nhau
    def main(image_path, embeddings_path, image_directory, save_folder, so_luong):
        # Tạo embedding cho ảnh cần tìm
        search_embedding = get_image_embedding(image_path)
        if search_embedding is None:
            return

        # Tải embeddings và tên ảnh đã lưu
        all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
        
        # Tìm và in ra các ảnh có độ tương đồng cao nhất và lưu ảnh
        search_similar_images(search_embedding, all_embeddings, image_names, image_directory, top_n=so_luong, save_folder=save_folder)

    # Chạy chương trình
    if __name__ == "__main__":
        main(image_path, embeddings_path, image_directory, save_folder, so_luong)
    count_nb+=1
# import os
# import cv2
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from model_lib.embedding import Embedding

# # Đường dẫn tới ảnh cần tìm kiếm
# image_path = "/home/dangph/Pictures/Screenshots/Screenshot from 2024-09-09 08-44-35.png"

# # Đường dẫn tới file lưu embeddings và thông tin các ảnh
# embeddings_path = '/mnt/sdb1/data_donghonuoc/clustering/embeddings/embeddings_images_meters_t6.npy'

# # Thư mục chứa các ảnh đã lưu embeddings
# image_directory = "/mnt/sdb1/data_donghonuoc/2024/06_img"

# # Thư mục để lưu các ảnh tương tự nhất
# save_folder = "/mnt/sdb1/data_donghonuoc/clustering/results/similar_images_results_1"

# # Bước 1: Tạo embedding từ ảnh cần tìm kiếm
# def get_image_embedding(image_path):
#     embedding = Embedding.getInstance()

#     # Đọc ảnh từ file
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Image {image_path} not found or failed to read.")
#         return None

#     # Chuyển đổi từ BGR sang RGB (nếu cần)
#     img = img[:, :, ::-1]
    
#     # Tạo embedding vector cho ảnh
#     vec = embedding([img], ["image_to_search"])
    
#     # Lấy embedding vector từ kết quả
#     return vec[0]['vector']

# # Bước 2: Tải danh sách các embeddings đã lưu và danh sách tên ảnh từ thư mục ảnh
# def load_saved_embeddings(embeddings_path, image_directory):
#     # Tải file embeddings, cho phép pickle
#     all_embeddings = np.load(embeddings_path, allow_pickle=True)
    
#     # Giả sử mỗi phần tử trong `all_embeddings` là một dict, ta cần lấy 'vector' từ mỗi phần tử
#     all_vectors = np.array([embedding['vector'] for embedding in all_embeddings])
    
#     # Lấy danh sách tên ảnh từ thư mục chứa ảnh
#     image_names = sorted([f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))])
    
#     # Kiểm tra độ dài của embeddings và image_names để đảm bảo khớp nhau
#     if len(all_vectors) != len(image_names):
#         print("Warning: The number of embeddings does not match the number of images.")
    
#     return all_vectors, image_names

# # Bước 3: Tính độ tương đồng và tìm kiếm các ảnh gần nhất
# def search_similar_images(search_embedding, all_embeddings, image_names, top_n=500, save_folder=None):
#     # Tính cosine similarity giữa ảnh cần tìm và các ảnh đã lưu
#     similarity_scores = cosine_similarity([search_embedding], all_embeddings)[0]
    
#     # Sắp xếp theo độ tương đồng giảm dần
#     sorted_indices = np.argsort(similarity_scores)[::-1]
    
#     # Đảm bảo thư mục lưu ảnh tồn tại
#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)

#     # In ra top N kết quả giống nhau nhất và lưu ảnh
#     print(f"Top {top_n} ảnh có độ tương đồng cao nhất:")
#     for i in range(top_n):
#         idx = sorted_indices[i]
#         image_name = image_names[idx]
#         similarity_score = similarity_scores[idx]

#         print(f"Ảnh: {image_name}, Similarity: {similarity_score:.4f}")

#         # Tạo đường dẫn đầy đủ tới ảnh
#         image_file_path = os.path.join(image_directory, image_name)

#         # Đọc ảnh từ file và lưu vào thư mục đích
#         if os.path.exists(image_file_path):  # Kiểm tra xem ảnh có tồn tại không
#             image_to_save = cv2.imread(image_file_path)  # Đọc ảnh gốc từ file
#             if image_to_save is not None:
#                 # Lưu ảnh với tên mới
#                 save_path = os.path.join(save_folder, f"similar_{i+1}_{similarity_score:.4f}.jpg")
#                 cv2.imwrite(save_path, image_to_save)
#                 print(f"Đã lưu ảnh: {save_path}")
#             else:
#                 print(f"Không thể đọc ảnh: {image_name}")
#         else:
#             print(f"Không tìm thấy ảnh: {image_file_path}")

# # Bước 4: Chạy các bước trên để tìm ảnh giống nhau
# def main(image_path, embeddings_path, image_directory, save_folder):
#     # Tạo embedding cho ảnh cần tìm
#     search_embedding = get_image_embedding(image_path)
#     if search_embedding is None:
#         return

#     # Tải embeddings và tên ảnh đã lưu
#     all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
    
#     # Tìm và in ra các ảnh có độ tương đồng cao nhất và lưu ảnh
#     search_similar_images(search_embedding, all_embeddings, image_names, top_n=500, save_folder=save_folder)

# # Chạy chương trình
# if __name__ == "__main__":
#     main(image_path, embeddings_path, image_directory, save_folder)
