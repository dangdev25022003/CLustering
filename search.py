# from tqdm import tqdm

# import os
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize
# from model_lib.embedding import Embedding
# import glob
# import shutil
# from PIL import Image
# a = ["4","5","6","7","8"]
# for i in a:
#     # Đường dẫn tới ảnh cần tìm kiếm
#     image_path = "/home/dangph/Pictures/Screenshots/Screenshot from 2024-09-19 10-29-27.png"

#     # Đường dẫn tới file lưu embeddings và thông tin các ảnh
#     embeddings_path = "/media/dangph/data-1tb/data_donghonuoc/clustering/embeddings/embeddings_images_meters_t"+i+".npy"

#     # Thư mục để lưu các ảnh tương tự nhất, thêm số đếm vào đường dẫn
#     save_folder = f"/home/dangph/Downloads/check_mặt_đồng hồ_full pipeline/3"

#     # Thư mục chứa ảnh cần xử lý
#     image_directory = f"/media/dangph/data-1tb/data_donghonuoc/2024/0"+i+"_img"



# # Bước 1: Tạo embedding từ ảnh cần tìm kiếm
# def get_image_embedding(image_path):
#     embedding = Embedding.getInstance()

#     try:
#         img = Image.open(image_path).convert('RGB')
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

#     img = np.array(img)
#     vec = embedding([img], ["image_to_search"])
#     return vec[0]['vector']

# def load_saved_embeddings(embeddings_path, image_directory):
#     all_embeddings = np.load(embeddings_path, allow_pickle=True)
#     image_names = [os.path.basename(f) for f in glob.glob(os.path.join(image_directory, '*.jpg'))]
#     all_vectors = np.array([embedding['vector'] for embedding in all_embeddings])
#     return all_vectors, image_names

# def search_similar_images(search_embedding, all_embeddings, image_names, top_n=200, save_folder=None):
#     similarity_scores = cosine_similarity([search_embedding], all_embeddings)[0]
#     sorted_indices = np.argsort(similarity_scores)[::-1]
#     print(similarity_scores)
#     # Đảm bảo thư mục lưu ảnh tồn tại
#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)

#     # Số lượng ảnh có thể có
#     num_images = len(image_names)
#     top_n = min(top_n, num_images)  # Đảm bảo không vượt quá số lượng ảnh có sẵn

#     print(f"Top {top_n} ảnh có độ tương đồng cao nhất:")
#     for idx in tqdm(sorted_indices[:top_n]):
#         if idx < len(image_names):  # Kiểm tra chỉ mục
#             image_name = image_names[idx]
#             similarity_score = similarity_scores[idx]
#             # print(f"Ảnh: {image_name}, Similarity: {similarity_score:.4f}")

#             image_file_path = os.path.join(image_directory, image_name)
#             if os.path.exists(image_file_path):
#                 save_path = os.path.join(save_folder, image_name)
#                 shutil.copy(image_file_path, save_path)
#                 # print(f"Đã sao chép ảnh: {save_path}")
#             else:
#                 print(f"Không tìm thấy ảnh: {image_file_path}")

# def main(image_path, embeddings_path, save_folder):
#     search_embedding = get_image_embedding(image_path)
#     if search_embedding is None:
#         return
#     all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
#     search_similar_images(search_embedding, all_embeddings, image_names, top_n=200, save_folder=save_folder)

# if __name__ == "__main__":
#     main(image_path, embeddings_path, save_folder)
#     #     folder_count_2 +=1
#     # folder_count += 1 




# from tqdm import tqdm

# import os
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize
# from model_lib.embedding import Embedding
# import glob
# import shutil
# from PIL import Image
# a = ["13106", "13107", "13108","13109","13110"]
# for i in a:
#     # Đường dẫn tới ảnh cần tìm kiếm
#     image_path = "/media/dangph/data/embedding_img_1/0001246.jpg"

#     # Đường dẫn tới file lưu embeddings và thông tin các ảnh
#     embeddings_path = "/media/dangph/data-1tb1/data_donghonuoc/clustering/embeddings/embedding_full_SKU_2.npy"

#     # Thư mục để lưu các ảnh tương tự nhất, thêm số đếm vào đường dẫn
#     save_folder = f"/media/dangph/data/clusters_3/{i}"

#     # Thư mục chứa ảnh cần xử lý
#     image_directory = f"/media/dangph/data/embedding/{i}"


#     # Bước 1: Tạo embedding từ ảnh cần tìm kiếm
#     def get_image_embedding(image_path):
#         embedding = Embedding.getInstance()

#         try:
#             img = Image.open(image_path).convert('RGB')
#         except Exception as e:
#             print(f"Error: {e}")
#             return None

#         img = np.array(img)
#         vec = embedding([img], ["image_to_search"])
#         return vec[0]['vector']

#     def normalize_embeddings(embeddings):
#         return normalize(embeddings, axis=1)

#     def load_saved_embeddings(embeddings_path, image_directory):
#         all_embeddings = np.load(embeddings_path, allow_pickle=True)
#         image_names = [os.path.basename(f) for f in glob.glob(os.path.join(image_directory, '*.jpg'))]
#         all_vectors = np.array([embedding['vector'] for embedding in all_embeddings])
        
#         # Normalize embeddings before returning
#         all_vectors = normalize_embeddings(all_vectors)
        
#         return all_vectors, image_names

#     def search_similar_images(search_embedding, all_embeddings, image_names, top_n=200, save_folder=None):
#         search_embedding = normalize_embeddings([search_embedding])[0]  # Normalize search embedding
#         similarity_scores = cosine_similarity([search_embedding], all_embeddings)[0]
#         sorted_indices = np.argsort(similarity_scores)[::-1]
#         # print(similarity_scores)
        
#         # Đảm bảo thư mục lưu ảnh tồn tại
#         if save_folder is not None:
#             os.makedirs(save_folder, exist_ok=True)

#         # Số lượng ảnh có thể có
#         num_images = len(image_names)
#         top_n = min(top_n, num_images)  # Đảm bảo không vượt quá số lượng ảnh có sẵn

#         # print(f"Top {top_n} ảnh có độ tương đồng cao nhất:")
#         count_number_in_folder = 0
#         for idx in tqdm(sorted_indices[:top_n]):
#             if idx < len(image_names):  # Kiểm tra chỉ mục
#                 image_name = image_names[idx]
#                 similarity_score = similarity_scores[idx]
#                 if similarity_score < 0.5:
#                     break
#                 else:
#                 # print(f"Ảnh: {image_name}, Similarity: {similarity_score:.4f}")
#                     image_file_path = os.path.join(image_directory, image_name)
#                     if os.path.exists(image_file_path):
#                         save_path = os.path.join(save_folder, image_name)
#                         shutil.copy(image_file_path, save_path)
#                         # print(f"Đã sao chép ảnh: {save_path}")
#                         count_number_in_folder +=1
#                     else:
#                         print(f"Không tìm thấy ảnh: {image_file_path}")
#         print(count_number_in_folder)

#     def main(image_path, embeddings_path, save_folder):
#         search_embedding = get_image_embedding(image_path)
#         if search_embedding is None:
#             return
#         all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
#         search_similar_images(search_embedding, all_embeddings, image_names, top_n=200, save_folder=save_folder)

#     if __name__ == "__main__":
#         main(image_path, embeddings_path, save_folder)


from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from model_lib.embedding import Embedding
import glob
import shutil
from PIL import Image

# List of folders to process
a = ["13106", "13112", "13128", "13146", "13153"]

# Path to the image to search for
image_path = "/media/dangph/data/embedding_img/1269260.jpg"

# Main folder to save similar images
save_folder = "/media/dangph/data/clusters_3"

# Function to create embedding from image
def get_image_embedding(image_path):
    embedding = Embedding.getInstance()
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: {e}")
        return None
    img = np.array(img)
    vec = embedding([img], ["image_to_search"])
    return vec[0]['vector']

# Normalize embeddings
def normalize_embeddings(embeddings):
    return normalize(embeddings, axis=1)

# Load embeddings from a file
def load_saved_embeddings(embeddings_path, image_directory):
    all_embeddings = np.load(embeddings_path, allow_pickle=True)
    image_names = [os.path.basename(f) for f in glob.glob(os.path.join(image_directory, '*.jpg'))]
    all_vectors = np.array([embedding['vector'] for embedding in all_embeddings])
    all_vectors = normalize_embeddings(all_vectors)
    return all_vectors, image_names

# Search for similar images
def search_similar_images(search_embedding, all_embeddings, image_names, top_n=100):
    search_embedding = normalize_embeddings([search_embedding])[0]
    similarity_scores = cosine_similarity([search_embedding], all_embeddings)[0]
    sorted_indices = np.argsort(similarity_scores)[::-1]
    return similarity_scores, sorted_indices

# Main function to handle all folders in 'a'
def main(image_path, save_folder):
    search_embedding = get_image_embedding(image_path)
    if search_embedding is None:
        return
    
    best_folder = None
    best_similarity_scores = []
    best_image_names = []
    max_similarity_across_folders = -1
    all_collected_images = []

    # Loop through each folder to find the folder with the highest max similarity score
    for i in a:
        embeddings_path = "/media/dangph/data-1tb1/data_donghonuoc/clustering/embeddings/embedding_full_SKU_2.npy"
        image_directory = f"/media/dangph/data/embedding_img/{i}/*"
        
        all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
        similarity_scores, sorted_indices = search_similar_images(search_embedding, all_embeddings, image_names, top_n=100)
        
        max_similarity = similarity_scores[sorted_indices[0]]
        print(f"Folder {i}: Max Similarity: {max_similarity:.4f}, Min Similarity: {similarity_scores[sorted_indices[-1]]:.4f}")
        
        # Update best folder if this folder has the highest max similarity score
        if max_similarity > max_similarity_across_folders:
            best_folder = i
            best_similarity_scores = similarity_scores
            best_image_names = image_names
            max_similarity_across_folders = max_similarity

    # Determine the threshold from the best folder (similarity score of the 500th image)
    sorted_indices = np.argsort(best_similarity_scores)[::-1]
    threshold_similarity = best_similarity_scores[sorted_indices[99]]
    
    print(f"\nBest Folder: {best_folder}, Max Similarity: {max_similarity_across_folders:.4f}, Threshold Similarity: {threshold_similarity:.4f}")
    
    # Save top 500 images from the best folder
    best_save_folder = os.path.join(save_folder, f"best_folder_{best_folder}")
    os.makedirs(best_save_folder, exist_ok=True)
    
    count_number_in_folder = 0
    for idx in tqdm(sorted_indices[:100]):
        if idx < len(best_image_names):
            image_name = best_image_names[idx]
            image_file_path = os.path.join(f"/media/dangph/data/clusters_3/0{best_folder}_img", image_name)
            if os.path.exists(image_file_path):
                save_path = os.path.join(best_save_folder, image_name)
                shutil.copy(image_file_path, save_path)
                count_number_in_folder += 1
                # Add to collected images
                all_collected_images.append((best_similarity_scores[idx], image_file_path))
    
    print(f"Saved {count_number_in_folder} images from the best folder.")
    
    # Now apply the threshold to other folders and collect images
    for i in a:
        if i == best_folder:
            continue
        
        embeddings_path = "/media/dangph/data-1tb1/data_donghonuoc/clustering/embeddings/embedding_full_SKU_2.npy"
        image_directory = f"/media/dangph/data/embedding_img/{i}/*"
        
        all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
        similarity_scores, sorted_indices = search_similar_images(search_embedding, all_embeddings, image_names, top_n=100)
        
        # Collect images that have similarity >= threshold_similarity
        for idx in sorted_indices:
            if similarity_scores[idx] >= threshold_similarity:
                image_name = image_names[idx]
                image_file_path = os.path.join(image_directory, image_name)
                if os.path.exists(image_file_path):
                    # Collect image and its similarity score
                    all_collected_images.append((similarity_scores[idx], image_file_path))
            else:
                break

    # Sort all collected images by similarity score in descending order
    all_collected_images = sorted(all_collected_images, key=lambda x: x[0], reverse=True)
    
    # Save the top 500 images across all folders
    final_save_folder = os.path.join(save_folder, "final_top_500_images")
    os.makedirs(final_save_folder, exist_ok=True)
    
    for i, (similarity_score, image_file_path) in enumerate(all_collected_images[:100]):
        save_path = os.path.join(final_save_folder, os.path.basename(image_file_path))
        shutil.copy(image_file_path, save_path)
    
    print(f"Saved the final top 500 images across all folders to {final_save_folder}.")

if __name__ == "__main__":
    main(image_path, save_folder)





# from tqdm import tqdm
# import os
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize
# from model_lib.embedding import Embedding
# import glob
# import shutil
# from PIL import Image

# # List of folders to process
# a = ["13106", "13107", "13108", "13109", "13110"]

# # Path to the image to search for
# image_path = "/media/dangph/data/clusters_2/13106/0070002.jpg"

# # Main folder to save similar images
# save_folder = "/media/dangph/data/clusters_2/13106_3"

# # Function to create embedding from image
# def get_image_embedding(image_path):
#     embedding = Embedding.getInstance()
#     try:
#         img = Image.open(image_path).convert('RGB')
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
#     img = np.array(img)
#     vec = embedding([img], ["image_to_search"])
#     return vec[0]['vector']

# # Normalize embeddings
# def normalize_embeddings(embeddings):
#     return normalize(embeddings, axis=1)

# # Load embeddings from a file
# def load_saved_embeddings(embeddings_path, image_directory):
#     all_embeddings = np.load(embeddings_path, allow_pickle=True)
#     image_names = [os.path.basename(f) for f in glob.glob(os.path.join(image_directory, '*.jpg'))]
#     all_vectors = np.array([embedding['vector'] for embedding in all_embeddings])
#     all_vectors = normalize_embeddings(all_vectors)
#     return all_vectors, image_names
# # Search for similar images
# def search_similar_images(search_embedding, all_embeddings, image_names, top_n=100):
#     search_embedding = normalize_embeddings([search_embedding])[0]
#     print(search_embedding.shape)
#     print(all_embeddings.shape)
    
#     similarity_scores = cosine_similarity([search_embedding], all_embeddings)[0]
#     sorted_indices = np.argsort(similarity_scores)[::-1]
#     return similarity_scores, sorted_indices

# # Main function to handle all folders in 'a'
# def main(image_path, save_folder):
#     search_embedding = get_image_embedding(image_path)
#     if search_embedding is None:
#         return
    
#     best_folder = None
#     best_similarity_scores = []
#     best_image_names = []
#     max_similarity_across_folders = -1
#     all_collected_images = []

#     # Loop through each folder to find the folder with the highest max similarity score
#     for i in a:
#         embeddings_path = f"/media/dangph/data-1tb1/data_donghonuoc/clustering/embeddings/embedding_full_SKU_2.npy"
#         image_directory = f"/media/dangph/data/clusters_2/{i}"
        
#         all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
#         similarity_scores, sorted_indices = search_similar_images(search_embedding, all_embeddings, image_names, top_n=100)
        
#         max_similarity = similarity_scores[sorted_indices[0]]
#         print(f"Folder {i}: Max Similarity: {max_similarity:.4f}, Min Similarity: {similarity_scores[sorted_indices[-1]]:.4f}")
        
#         # Update best folder if this folder has the highest max similarity score
#         if max_similarity > max_similarity_across_folders:
#             best_folder = i
#             best_similarity_scores = similarity_scores
#             best_image_names = image_names
#             max_similarity_across_folders = max_similarity

#     # Determine the threshold from the best folder (similarity score of the 500th image)
#     sorted_indices = np.argsort(best_similarity_scores)[::-1]
#     threshold_similarity = best_similarity_scores[sorted_indices[99]]
    
#     print(f"\nBest Folder: {best_folder}, Max Similarity: {max_similarity_across_folders:.4f}, Threshold Similarity: {threshold_similarity:.4f}")
    
#     # Save top 500 images from the best folder
#     best_save_folder = os.path.join(save_folder, f"best_folder_{best_folder}")
#     os.makedirs(best_save_folder, exist_ok=True)
    
#     count_number_in_folder = 0
#     for idx in tqdm(sorted_indices[:100]):
#         if idx < len(best_image_names):
#             image_name = best_image_names[idx]
#             image_file_path = os.path.join(f"/media/dangph/data/clusters_2/0{best_folder}_img", image_name)
#             if os.path.exists(image_file_path):
#                 save_path = os.path.join(best_save_folder, image_name)
#                 shutil.copy(image_file_path, save_path)
#                 count_number_in_folder += 1
#                 # Add to collected images
#                 all_collected_images.append((best_similarity_scores[idx], image_file_path))
    
#     print(f"Saved {count_number_in_folder} images from the best folder.")
    
#     # Now apply the threshold to other folders and collect images
#     for i in a:
#         if i == best_folder:
#             continue
        
#         embeddings_path = f"/media/dangph/data-1tb1/data_donghonuoc/clustering/embeddings/embedding_full_SKU_2.npy"
#         image_directory = f"/media/dangph/data/clusters_2/{i}"
        
#         all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
#         similarity_scores, sorted_indices = search_similar_images(search_embedding, all_embeddings, image_names, top_n=500)
        
#         # Collect images that have similarity >= threshold_similarity
#         for idx in sorted_indices:
#             if similarity_scores[idx] >= threshold_similarity:
#                 image_name = image_names[idx]
#                 image_file_path = os.path.join(image_directory, image_name)
#                 if os.path.exists(image_file_path):
#                     # Collect image and its similarity score
#                     all_collected_images.append((similarity_scores[idx], image_file_path))
#             else:
#                 break

#     # Sort all collected images by similarity score in descending order
#     all_collected_images = sorted(all_collected_images, key=lambda x: x[0], reverse=True)
    
#     # Save the top 500 images across all folders
#     final_save_folder = os.path.join(save_folder, "final_top_500_images")
#     os.makedirs(final_save_folder, exist_ok=True)
    
#     for i, (similarity_score, image_file_path) in enumerate(all_collected_images[:100]):
#         save_path = os.path.join(final_save_folder, os.path.basename(image_file_path))
#         shutil.copy(image_file_path, save_path)
    
#     print(f"Saved the final top 500 images across all folders to {final_save_folder}.")

# if __name__ == "__main__":
#     main(image_path, save_folder)
