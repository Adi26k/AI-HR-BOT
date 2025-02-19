import numpy as np
from scipy.spatial.distance import cosine
import hnswlib  # Ensure hnswlib is installed

class JobMatching:
    def __init__(self, job_postings, use_hnsw=True):
        """
        Initializes the JobMatching system.
        :param job_postings: List of job postings with vector embeddings.
        :param use_hnsw: Whether to use HNSW for approximate nearest neighbor search.
        """
        self.job_postings = job_postings
        self.use_hnsw = use_hnsw
        self.dim = len(job_postings[0]['vector_embedding']) if job_postings else 0  # Get embedding dimension
        
        if use_hnsw and self.dim > 0:
            self.hnsw_index = self.build_hnsw_index()
        else:
            self.hnsw_index = None
    
    def build_hnsw_index(self):
        """Builds an HNSW index for fast nearest neighbor search."""
        max_elements = len(self.job_postings)
        hnsw_index = hnswlib.Index(space='cosine', dim=self.dim)

        # Optimized values for stability
        hnsw_index.init_index(max_elements=max_elements, ef_construction=500, M=64)
        
        for i, job in enumerate(self.job_postings):
            hnsw_index.add_items(np.array(job['vector_embedding'], dtype=np.float32), i)

        hnsw_index.set_ef(600)  # Increase ef for better recall
        return hnsw_index
    
    def find_best_jobs(self, application_embedding, k=5):
        """
        Finds the top k job postings based on vector similarity.
        :param application_embedding: Vector embedding of the job application.
        :param k: Number of top job postings to return.
        """
        application_embedding = np.array(application_embedding, dtype=np.float32).reshape(1, -1)

        if self.use_hnsw and self.hnsw_index is not None:
            return self.hnsw_search(application_embedding, k)
        else:
            return self.exhaustive_knn_search(application_embedding, k)
    
    def exhaustive_knn_search(self, application_embedding, k):
        """Performs an exhaustive KNN search by comparing all job vectors."""
        similarities = []

        for job in self.job_postings:
            score = 1 - cosine(application_embedding, job['vector_embedding'])
            similarities.append((job, score))

        sorted_jobs = sorted(similarities, key=lambda x: x[1], reverse=True)
        return sorted_jobs[:k]
    
    def hnsw_search(self, application_embedding, k):
        """Performs an approximate nearest neighbor search using HNSW."""
        
        # Debugging: Ensure correct shape and type
        print(f"🔹 Application Embedding Type: {type(application_embedding)}, Shape: {application_embedding.shape}")
        print(f"🔹 HNSW Index Size: {self.hnsw_index.get_current_count()}")

        try:
            labels, distances = self.hnsw_index.knn_query(application_embedding, k=k)
            results = [(self.job_postings[int(idx)], 1 - dist) for idx, dist in zip(labels[0].tolist(), distances[0].tolist())]
            return results
        except RuntimeError as e:
            print(f"⚠️ HNSW search failed: {e}. Trying to rebuild index...")
            
            # Rebuild the index if it's empty or corrupted
            if self.hnsw_index.get_current_count() == 0:
                self.hnsw_index = self.build_hnsw_index()
                
            raise RuntimeError(f"HNSW search failed after rebuilding: {e}. Try reducing M or increasing ef.")  
