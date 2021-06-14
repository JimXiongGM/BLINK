


from index.faiss_indexer import (
    DenseFlatIndexer,
    DenseHNSWFlatIndexer,
    DenseIVFFlatIndexer,
)
indexer = DenseHNSWFlatIndexer(1)
indexer.deserialize_from("blink_elq_data/elq/models/faiss_hnsw_index.pkl")

x = 2
