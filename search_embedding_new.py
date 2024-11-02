import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from model_lib.embedding import Embedding
from PIL import Image
import glob
import shutil

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

# Load embeddings from a directory
def load_embeddings_from_directory(image_directory):
    image_paths = glob.glob(os.path.join(image_directory, '*.jpg'))
    embeddings = []
    for img_path in image_paths:
        embedding = get_image_embedding(img_path)
        if embedding is not None:
            embeddings.append(embedding)
    return np.array(embeddings), image_paths

# Perform clustering using cosine similarity
def cluster_images(image_directory, n_clusters=5):
    embeddings, image_paths = load_embeddings_from_directory(image_directory)
    if len(embeddings) == 0:
        print("No valid embeddings found.")
        return
    
    # Normalize embeddings
    normalized_embeddings = normalize_embeddings(embeddings)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(normalized_embeddings)

    # Use KMeans to cluster based on similarity
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(similarity_matrix)

    return labels, image_paths

# Main function
def main(image_directory, save_directory, n_clusters=5):
    labels, image_paths = cluster_images(image_directory, n_clusters)

    # Create directories for each cluster
    for cluster_id in range(n_clusters):
        os.makedirs(os.path.join(save_directory, f"cluster_{cluster_id}"), exist_ok=True)

    # Save images to corresponding cluster directories
    for label, img_path in zip(labels, image_paths):
        destination = os.path.join(save_directory, f"cluster_{label}", os.path.basename(img_path))
        shutil.copy(img_path, destination)
    
    print(f"Images have been clustered into {n_clusters} groups and saved in {save_directory}.")

if __name__ == "__main__":
    image_directory = "/media/dangph/data/embedding_img"  # Thay đổi đường dẫn đến thư mục chứa hình ảnh của bạn
    save_directory = "clus/media/dangph/data/embedding_1"  # Thay đổi đường dẫn đến thư mục lưu ảnh sau khi phân cụm
    n_clusters = 5  # Số lượng cụm mong muốn
    main(image_directory, save_directory)
