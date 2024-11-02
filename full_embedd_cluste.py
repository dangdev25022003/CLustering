# cho 1 folder

# import os
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# from PIL import Image
# import shutil
# from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# from model_lib.embedding import Embedding
# import glob
# from sklearn.preprocessing import normalize

# # Define paths
# IMAGE_DIR = '/media/dangph/data/embedding/13155'  # Update to your image directory
# OUTPUT_DIR = '/media/dangph/data/clusters_2'
# SAVE_FOLDER = '/media/dangph/data/clusters_3'

# # Ensure output directories exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(SAVE_FOLDER, exist_ok=True)

# # Load and preprocess images
# def load_images(image_dir):
#     images, all_vectors = [], {}
#     for filename in tqdm(os.listdir(image_dir), desc="Loading images"):
#         if filename.lower().endswith(('.jpg', '.png')):
#             image_path = os.path.join(image_dir, filename)
#             try:
#                 image = Image.open(image_path).convert('RGB')
#                 image = image.resize((64, 64))  # Resize to reduce dimensionality
#                 image_array = np.array(image).flatten()  # Flatten to 1D array
#                 images.append(image_array)
#                 all_vectors[image_path] = filename
#             except Exception as e:
#                 print(f"Error loading {filename}: {e}")
#     return np.array(images), all_vectors

# # Load images
# images, all_vectors = load_images(IMAGE_DIR)

# # Step 1: Dimensionality Reduction with t-SNE
# tsne = TSNE(n_components=2, perplexity=5, random_state=42)
# embeddings_tsne = tsne.fit_transform(images)

# # Step 2: Clustering with KMeans (Fixed to 5 clusters)
# num_clusters = 5  # Set the number of clusters to 5
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# labels = kmeans.fit_predict(embeddings_tsne)

# # Create cluster directories
# for label in range(num_clusters):
#     label_dir = os.path.join(OUTPUT_DIR, f'cluster_{label}')
#     os.makedirs(label_dir, exist_ok=True)

# # Copy images into respective cluster directories
# for idx, (image_path, label) in tqdm(enumerate(zip(all_vectors.keys(), labels)), 
#                                      total=len(all_vectors), desc="Copying images to clusters"):
#     dst_path = os.path.join(OUTPUT_DIR, f'cluster_{label}', os.path.basename(image_path))
#     shutil.copy(image_path, dst_path)

# print("Clustering completed.")

# # Image embedding function
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

# # Calculate embeddings for all images in the directory
# def load_image_embeddings(image_directory):
#     embedding = Embedding.getInstance()
#     all_vectors = []
#     image_names = []

#     for img_file in tqdm(glob.glob(os.path.join(image_directory, '*.jpg'))):
#         try:
#             img = Image.open(img_file).convert('RGB')
#             img_array = np.array(img)
#             vec = embedding([img_array], [img_file])
#             all_vectors.append(vec[0]['vector'])
#             image_names.append(os.path.basename(img_file))
#         except Exception as e:
#             print(f"Error processing image {img_file}: {e}")

#     # Normalize embeddings
#     all_vectors = normalize_embeddings(np.array(all_vectors))
#     return all_vectors, image_names

# # Search for similar images
# def search_similar_images(search_embedding, all_embeddings, image_names, top_n=15, save_folder=None):
#     search_embedding = normalize_embeddings([search_embedding])[0]  # Normalize search embedding
#     cosine_scores = cosine_similarity([search_embedding], all_embeddings)[0]
#     euclidean_scores = euclidean_distances([search_embedding], all_embeddings)[0]

#     # Combine scores
#     combined_scores = cosine_scores - euclidean_scores
#     sorted_indices = np.argsort(combined_scores)[::-1]

#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)

#     print(f"Top {top_n} most similar images:")
#     for idx in sorted_indices[:top_n]:
#         image_name = image_names[idx]
#         similarity_score = combined_scores[idx]
#         print(f"Image: {image_name}, Similarity: {similarity_score:.4f}")

#         # Update the path to ensure it's pointing to the correct directory
#         image_file_path = os.path.join(IMAGE_DIR, image_name)  # Change IMAGE_DIR to match your actual directory
#         print(f"Looking for image at: {image_file_path}")  # Debugging line
#         if os.path.exists(image_file_path):
#             save_path = os.path.join(save_folder, image_name)
#             shutil.copy(image_file_path, save_path)
#         else:
#             print(f"Image not found: {image_file_path}")

# def search_across_clusters(image_path_to_search, output_dir, save_folder):
#     # Load embeddings and image names from each cluster
#     cluster_embeddings = {}
#     cluster_image_names = {}
#     cluster_similarities = {}

#     for label in range(num_clusters):
#         cluster_dir = os.path.join(output_dir, f'cluster_{label}')
#         embeddings, image_names = load_image_embeddings(cluster_dir)
#         cluster_embeddings[label] = embeddings
#         cluster_image_names[label] = image_names

#         # Get search embedding for the current cluster's first image
#         search_embedding = get_image_embedding(image_path_to_search)

#         # Calculate the average similarity for this cluster
#         avg_similarity = np.mean(cosine_similarity([search_embedding], embeddings)[0])
#         cluster_similarities[label] = avg_similarity

#     # Determine the cluster with the lowest average similarity
#     lowest_similarity_cluster = min(cluster_similarities, key=cluster_similarities.get)
#     print(f"Lowest similarity cluster: {lowest_similarity_cluster} with similarity {cluster_similarities[lowest_similarity_cluster]:.4f}")

#     # Print the name of the folder being removed
#     lowest_similarity_cluster_name = f'cluster_{lowest_similarity_cluster}'
#     print(f"Removing cluster folder: {lowest_similarity_cluster_name}")

#     # Remove the lowest similarity cluster from further processing
#     del cluster_embeddings[lowest_similarity_cluster]
#     del cluster_image_names[lowest_similarity_cluster]

#     # Search for similar images in the remaining clusters
#     for label, embeddings in cluster_embeddings.items():
#         image_names = cluster_image_names[label]
#         search_similar_images(search_embedding, embeddings, image_names, top_n=15, save_folder=save_folder)

# if __name__ == "__main__":
#     # Load embeddings and image names from the clusters
#     cluster_embeddings = {}
#     cluster_image_names = {}

#     for label in range(num_clusters):
#         cluster_dir = os.path.join(OUTPUT_DIR, f'cluster_{label}')
#         embeddings, image_names = load_image_embeddings(cluster_dir)
#         cluster_embeddings[label] = embeddings
#         cluster_image_names[label] = image_names

#     # Get the search embedding from the first image in the largest cluster
#     largest_cluster_label = max(cluster_embeddings.keys(), key=lambda k: len(cluster_image_names[k]))
#     image_path_to_search = os.path.join(OUTPUT_DIR, f'cluster_{largest_cluster_label}', cluster_image_names[largest_cluster_label][0])

#     # Search across clusters
#     search_across_clusters(image_path_to_search, OUTPUT_DIR, SAVE_FOLDER)



# cho nhiều folder



import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from PIL import Image
import shutil
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from model_lib.embedding import Embedding
import glob
from sklearn.preprocessing import normalize

# Define main directories
MAIN_DIR = '/media/dangph/data/embedding'  # Update this to your main directory path
RESULTS_DIR_2 = '/media/dangph/data/clusters_2'  # New directory for cluster results
RESULTS_DIR_3 = '/media/dangph/data/clusters_3'  # New directory for similar images results
MIN_FILES = 100  # Minimum number of files required to process a directory

# Create results directories
os.makedirs(RESULTS_DIR_2, exist_ok=True)
os.makedirs(RESULTS_DIR_3, exist_ok=True)

def count_image_files(directory):
    """Count the number of image files in a directory"""
    image_count = sum(1 for f in os.listdir(directory) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return image_count

def process_directory(parent_dir):
    # Check if directory has enough files
    file_count = count_image_files(parent_dir)
    if file_count < MIN_FILES:
        print(f"Skipping {parent_dir}: Only {file_count} files found (minimum {MIN_FILES} required)")
        return

    # Define paths for current directory
    image_dir = parent_dir
    dir_name = os.path.basename(parent_dir)
    output_dir = os.path.join(RESULTS_DIR_2, dir_name)
    save_folder = os.path.join(RESULTS_DIR_3, dir_name)

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    print(f"\nProcessing directory: {parent_dir}")
    print(f"Number of files: {file_count}")
    print(f"Output directory: {output_dir}")
    print(f"Save folder: {save_folder}")

    # Load and preprocess images
    def load_images(image_dir):
        images, all_vectors = [], {}
        for filename in tqdm(os.listdir(image_dir), desc="Loading images"):
            if filename.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(image_dir, filename)
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((64, 64))
                    image_array = np.array(image).flatten()
                    images.append(image_array)
                    all_vectors[image_path] = filename
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return np.array(images), all_vectors

    images, all_vectors = load_images(image_dir)
    
    if len(images) == 0:
        print(f"No valid images found in {image_dir}. Skipping directory.")
        return

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    embeddings_tsne = tsne.fit_transform(images)

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_tsne)

    for label in range(num_clusters):
        label_dir = os.path.join(output_dir, f'cluster_{label}')
        os.makedirs(label_dir, exist_ok=True)

    for idx, (image_path, label) in tqdm(enumerate(zip(all_vectors.keys(), labels)), 
                                       total=len(all_vectors), desc="Copying images to clusters"):
        dst_path = os.path.join(output_dir, f'cluster_{label}', os.path.basename(image_path))
        shutil.copy(image_path, dst_path)

    print(f"Clustering completed for {parent_dir}")

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

    def normalize_embeddings(embeddings):
        return normalize(embeddings, axis=1)

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

        if len(all_vectors) == 0:
            return None, None

        all_vectors = normalize_embeddings(np.array(all_vectors))
        return all_vectors, image_names

    def search_similar_images(search_embedding, all_embeddings, image_names, top_n=15, save_folder=None):
        search_embedding = normalize_embeddings([search_embedding])[0]
        cosine_scores = cosine_similarity([search_embedding], all_embeddings)[0]
        euclidean_scores = euclidean_distances([search_embedding], all_embeddings)[0]

        combined_scores = cosine_scores - euclidean_scores
        sorted_indices = np.argsort(combined_scores)[::-1]

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)

        print(f"Top {top_n} most similar images:")
        for idx in sorted_indices[:top_n]:
            image_name = image_names[idx]
            similarity_score = combined_scores[idx]
            print(f"Image: {image_name}, Similarity: {similarity_score:.4f}")

            image_file_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_file_path):
                save_path = os.path.join(save_folder, image_name)
                shutil.copy(image_file_path, save_path)
            else:
                print(f"Image not found: {image_file_path}")

    def search_across_clusters(image_path_to_search, output_dir, save_folder):
        cluster_embeddings = {}
        cluster_image_names = {}
        cluster_similarities = {}

        for label in range(num_clusters):
            cluster_dir = os.path.join(output_dir, f'cluster_{label}')
            embeddings, image_names = load_image_embeddings(cluster_dir)
            
            if embeddings is None:
                continue
                
            cluster_embeddings[label] = embeddings
            cluster_image_names[label] = image_names

            search_embedding = get_image_embedding(image_path_to_search)
            avg_similarity = np.mean(cosine_similarity([search_embedding], embeddings)[0])
            cluster_similarities[label] = avg_similarity

        if not cluster_similarities:
            print("No valid clusters found. Skipping similarity analysis.")
            return

        lowest_similarity_cluster = min(cluster_similarities, key=cluster_similarities.get)
        print(f"Lowest similarity cluster: {lowest_similarity_cluster} with similarity {cluster_similarities[lowest_similarity_cluster]:.4f}")

        lowest_similarity_cluster_name = f'cluster_{lowest_similarity_cluster}'
        print(f"Removing cluster folder: {lowest_similarity_cluster_name}")

        del cluster_embeddings[lowest_similarity_cluster]
        del cluster_image_names[lowest_similarity_cluster]

        for label, embeddings in cluster_embeddings.items():
            image_names = cluster_image_names[label]
            search_similar_images(search_embedding, embeddings, image_names, top_n=15, save_folder=save_folder)

    cluster_embeddings = {}
    cluster_image_names = {}

    for label in range(num_clusters):
        cluster_dir = os.path.join(output_dir, f'cluster_{label}')
        embeddings, image_names = load_image_embeddings(cluster_dir)
        
        if embeddings is None:
            continue
            
        cluster_embeddings[label] = embeddings
        cluster_image_names[label] = image_names

    if cluster_embeddings:
        largest_cluster_label = max(cluster_embeddings.keys(), key=lambda k: len(cluster_image_names[k]))
        image_path_to_search = os.path.join(output_dir, f'cluster_{largest_cluster_label}', 
                                          cluster_image_names[largest_cluster_label][0])

        search_across_clusters(image_path_to_search, output_dir, save_folder)
    else:
        print("No valid clusters found. Skipping search across clusters.")

def main():
    subdirs = [d for d in glob.glob(os.path.join(MAIN_DIR, '*')) if os.path.isdir(d)]
    
    print(f"Found {len(subdirs)} directories to process")
    
    for subdir in subdirs:
        try:
            print(f"\nStarting processing for directory: {subdir}")
            process_directory(subdir)
            print(f"Completed processing for directory: {subdir}")
        except Exception as e:
            print(f"Error processing directory {subdir}: {e}")
            continue

if __name__ == "__main__":
    main()