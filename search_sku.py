# from tqdm import tqdm
# import os
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from model_lib.embedding import Embedding
# import glob
# import shutil
# from PIL import Image

# # Đường dẫn tới ảnh cần tìm kiếm
# image_path = "/media/dangph/data/embedding/13146/0720655.jpg"

# # Đường dẫn tới file lưu embeddings và thông tin các ảnh
# embeddings_path = "/media/dangph/data-1tb1/data_donghonuoc/clustering/embeddings/embedding_full_SKU_2.npy"

# # Thư mục chứa ảnh cần xử lý
# image_directory = "/media/dangph/data/embedding/13146"

# # Thư mục để lưu các ảnh tương tự nhất
# save_folder = "/media/dangph/data/clusters_3"

# def get_image_embedding(image_path):
#     embedding = Embedding.getInstance()
#     try:
#         img = Image.open(image_path).convert('RGB')
#     except Exception as e:
#         print(f"Error opening image: {e}")
#         return None

#     img = np.array(img)
#     vec = embedding([img], ["image_to_search"])
#     return vec[0]['vector']

# def load_saved_embeddings(embeddings_path, image_directory):
#     all_embeddings = np.load(embeddings_path, allow_pickle=True)
#     image_names = [os.path.basename(f) for f in glob.glob(os.path.join(image_directory, '*.jpg'))]

#     # Ensure lengths match
#     if len(all_embeddings) != len(image_names):
#         print(f"Warning: Mismatch in counts - Embeddings: {len(all_embeddings)}, Images: {len(image_names)}")
#         min_len = min(len(all_embeddings), len(image_names))
#         all_embeddings = all_embeddings[:min_len]
#         image_names = image_names[:min_len]

#     all_vectors = np.array([embedding['vector'] for embedding in all_embeddings])
#     return all_vectors, image_names

# def search_similar_images(search_embedding, all_embeddings, image_names, top_n=100, save_folder=None):
#     similarity_scores = cosine_similarity([search_embedding], all_embeddings)[0]
#     sorted_indices = np.argsort(similarity_scores)[::-1]  # Sắp xếp giảm dần

#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)

#     top_n = min(top_n, len(image_names))  # Đảm bảo không vượt quá số lượng ảnh có sẵn

#     print(f"Top {top_n} ảnh có độ tương đồng cao nhất:")
#     for idx in tqdm(sorted_indices[:top_n]):
#         if idx >= len(image_names):  # Ensure index is valid
#             print(f"Index {idx} out of range, skipping.")
#             continue

#         image_name = image_names[idx]
#         similarity_score = similarity_scores[idx]
#         print(f"Ảnh: {image_name}, Similarity: {similarity_score:.4f}")

#         image_file_path = os.path.join(image_directory, image_name)
#         if os.path.exists(image_file_path):
#             save_path = os.path.join(save_folder, image_name)
#             shutil.copy(image_file_path, save_path)
#         else:
#             print(f"Không tìm thấy ảnh: {image_file_path}")

# def main(image_path, embeddings_path, image_directory, save_folder):
#     search_embedding = get_image_embedding(image_path)
#     if search_embedding is None:
#         return

#     all_embeddings, image_names = load_saved_embeddings(embeddings_path, image_directory)
#     search_similar_images(search_embedding, all_embeddings, image_names, top_n=100, save_folder=save_folder)

# if __name__ == "__main__":
#     main(image_path, embeddings_path, image_directory, save_folder)




# from tqdm import tqdm
# import os
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# from model_lib.embedding import Embedding
# import glob
# import shutil
# from PIL import Image
# from sklearn.preprocessing import normalize

# # Đường dẫn tới ảnh cần tìm kiếm
# image_path = "/media/dangph/data/embedding/13146/0720655.jpg"

# # Đường dẫn tới file lưu embeddings và thông tin các ảnh
# embeddings_path = "/media/dangph/data-1tb1/data_donghonuoc/clustering/embeddings/embedding_full_SKU_2.npy"

# # Thư mục chứa ảnh cần xử lý
# image_directory = "/media/dangph/data/embedding/13146"

# # Thư mục để lưu các ảnh tương tự nhất
# save_folder = "/media/dangph/data/clusters_3"

# def get_image_embedding(image_path):
#     embedding = Embedding.getInstance()
#     try:
#         img = Image.open(image_path).convert('RGB')
#     except Exception as e:
#         print(f"Error opening image: {e}")
#         return None

#     img = np.array(img)
#     vec = embedding([img], ["image_to_search"])
#     return vec[0]['vector']

# # Normalize embeddings
# def normalize_embeddings(embeddings):
#     return normalize(embeddings, axis=1)

# # Load saved embeddings and image names from a .npy file
# def load_saved_embeddings(embeddings_path):
#     all_embeddings = np.load(embeddings_path, allow_pickle=True)
#     all_vectors = np.array([embedding['vector'] for embedding in all_embeddings])
    
#     # Normalize embeddings
#     all_vectors = normalize_embeddings(all_vectors)
    
#     return all_vectors

# # Load image names from the directory
# def load_image_names(image_directory):
#     return [os.path.basename(f) for f in glob.glob(os.path.join(image_directory, '*.jpg'))]

# # Search for similar images
# def search_similar_images(search_embedding, all_embeddings, image_names, top_n=10, save_folder=None):
#     search_embedding = normalize_embeddings([search_embedding])[0]  # Normalize search embedding
#     cosine_scores = cosine_similarity([search_embedding], all_embeddings)[0]
#     euclidean_scores = euclidean_distances([search_embedding], all_embeddings)[0]

#     # Kết hợp các điểm số
#     combined_scores = cosine_scores - euclidean_scores

#     sorted_indices = np.argsort(combined_scores)[::-1]

#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)

#     # In số lượng ảnh và embeddings
#     num_images_available = len(image_names)
#     num_embeddings = len(all_embeddings)
#     print(f"Số lượng ảnh có sẵn: {num_images_available}, Số lượng embeddings: {num_embeddings}")

#     print(f"Top {top_n} ảnh có độ tương đồng cao nhất:")
#     valid_indices = [idx for idx in sorted_indices if idx < num_images_available]  # Lọc chỉ số hợp lệ
#     top_indices = valid_indices[:top_n]  # Lấy top N chỉ số hợp lệ

#     for idx in top_indices:
#         image_name = image_names[idx]
#         similarity_score = combined_scores[idx]
#         print(f"Ảnh: {image_name}, Similarity: {similarity_score:.4f}")

#         image_file_path = os.path.join(image_directory, image_name)
#         if os.path.exists(image_file_path):
#             save_path = os.path.join(save_folder, image_name)
#             shutil.copy(image_file_path, save_path)
#         else:
#             print(f"Không tìm thấy ảnh: {image_file_path}")



# def main(image_path, embeddings_path, image_directory, save_folder):
#     search_embedding = get_image_embedding(image_path)
#     if search_embedding is None:
#         return

#     all_embeddings = load_saved_embeddings(embeddings_path)
#     image_names = load_image_names(image_directory)
#     search_similar_images(search_embedding, all_embeddings, image_names, top_n=50, save_folder=save_folder)

# if __name__ == "__main__":
#     main(image_path, embeddings_path, image_directory, save_folder)


from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from model_lib.embedding import Embedding
import glob
import shutil
from PIL import Image
from sklearn.preprocessing import normalize

# Đường dẫn tới ảnh cần tìm kiếm
image_path = "/media/dangph/data/embedding/13146/0720655.jpg"

# Thư mục chứa ảnh cần xử lý
image_directory = "/media/dangph/data/clusters_2/cluster_1"

# Thư mục để lưu các ảnh tương tự nhất
save_folder = "/media/dangph/data/clusters_3"

def get_image_embedding(image_path):
    embedding = Embedding.getInstance()
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    img = np.array(img)
    vec = embedding([img], ["image_to_search"])
    return vec[0]['vector']

# Normalize embeddings
def normalize_embeddings(embeddings):
    return normalize(embeddings, axis=1)

# Tính toán embedding cho tất cả ảnh trong thư mục
def load_image_embeddings(image_directory):
    embedding = Embedding.getInstance()
    all_vectors = []
    image_names = []

    for img_file in tqdm(glob.glob(os.path.join(image_directory, '*.jpg'))):
        try:
            img = Image.open(img_file).convert('RGB')
            img_array = np.array(img)
            vec = embedding([img_array], [img_file])
            all_vectors.append(vec[0]['vector'])
            image_names.append(os.path.basename(img_file))
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")

    # Normalize embeddings
    all_vectors = normalize_embeddings(np.array(all_vectors))
    return all_vectors, image_names

# Search for similar images
def search_similar_images(search_embedding, all_embeddings, image_names, top_n=5, save_folder=None):
    search_embedding = normalize_embeddings([search_embedding])[0]  # Normalize search embedding
    cosine_scores = cosine_similarity([search_embedding], all_embeddings)[0]
    euclidean_scores = euclidean_distances([search_embedding], all_embeddings)[0]

    # Kết hợp các điểm số
    combined_scores = cosine_scores - euclidean_scores

    sorted_indices = np.argsort(combined_scores)[::-1]

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    print(f"Top {top_n} ảnh có độ tương đồng cao nhất:")
    for idx in sorted_indices[:top_n]:
        image_name = image_names[idx]
        similarity_score = combined_scores[idx]
        print(f"Ảnh: {image_name}, Similarity: {similarity_score:.4f}")

        image_file_path = os.path.join(image_directory, image_name)
        if os.path.exists(image_file_path):
            save_path = os.path.join(save_folder, image_name)
            shutil.copy(image_file_path, save_path)
        else:
            print(f"Không tìm thấy ảnh: {image_file_path}")

def main(image_path, image_directory, save_folder):
    search_embedding = get_image_embedding(image_path)
    if search_embedding is None:
        return

    all_embeddings, image_names = load_image_embeddings(image_directory)
    search_similar_images(search_embedding, all_embeddings, image_names, top_n=5, save_folder=save_folder)

if __name__ == "__main__":
    main(image_path, image_directory, save_folder)
