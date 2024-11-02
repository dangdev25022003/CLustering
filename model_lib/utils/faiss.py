import faiss
import numpy as np


class FAISSLookupIndex():
    def __init__(self, collection_name, data, exact=False, cpu=False, type='IP'):
        self.times_queried = 0
        self.exact = exact

        # creating the index
        self.index = None
        self.cpu_index_flat = None
        self.type = type
        self.cpu = cpu

        self.build_index(collection_name, data)

    def _build_approximate_index(self, collection_name, data: np.ndarray):
        dimensionality = data.shape[1]
        nlist = 100 if data.shape[0] > 100 else 2

        if self.type in {'L2'}:
            quantizer = faiss.IndexFlatL2(dimensionality)
            cpu_index_flat = faiss.IndexIVFFlat(quantizer, dimensionality, nlist, faiss.METRIC_L2)
        else:
            quantizer = faiss.IndexFlatIP(dimensionality)
            cpu_index_flat = faiss.IndexIVFFlat(quantizer, dimensionality, nlist)

        if not self.cpu:
            gpu_index_ivf = faiss.index_cpu_to_gpu(self.resource, 0, cpu_index_flat)
            gpu_index_ivf.train(data)
            gpu_index_ivf.add(data)
            self.index = gpu_index_ivf
        else:
            cpu_index_flat.train(data)
            cpu_index_flat.add(data)
            self.index = cpu_index_flat

        faiss.write_index(self.index, collection_name + '.index')
        

    def _build_exact_index(self, collection_name, data: np.ndarray):
        dimensionality = data.shape[1]

        if self.type in {'L2'}:
            self.cpu_index_flat = faiss.IndexFlatL2(dimensionality)
        else:
            self.cpu_index_flat = faiss.IndexFlatIP(dimensionality)

        if not self.cpu:
            resource = faiss.StandardGpuResources() 
            self.index = faiss.index_cpu_to_gpu(resource, 0, self.cpu_index_flat)
        else:
            self.index = self.cpu_index_flat
        self.index.add(data)

        faiss.write_index(self.index, collection_name + '.index')

    def query(self, collection_name,
                data: np.ndarray,
                k: int = 11,
                is_training: bool = False):
        _, neighbour_indices = self.index.search(data, k)
        self.times_queried += 1 if is_training else 0
        return neighbour_indices

    def build_index(self, collection_name, data: np.ndarray):
        self.times_queried = 0

        if self.exact:
            self._build_exact_index(collection_name, data)
        else:
            self._build_approximate_index(collection_name, data)

    def add_vector(self, collection_name, data: np.ndarray):
        index = faiss.read_index(collection_name + '.index')
        id_start = index.ntotal - 1
        id_end = id_start + data.shape[0]
        index.add_with_ids(data, np.arange(id_start, id_end))
        faiss.write_index(self.index, collection_name + '.index')

    def search(self, collection_name, data: np.ndarray, k=1):
        index = faiss.read_index(collection_name + '.index')
        distances, result_ids = index.search(data, k)
        return distances, result_ids

