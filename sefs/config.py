import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(BASE_DIR, "root_folder")

SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

EMBEDDING_MODEL = "all-mpnet-base-v2"


SIMILARITY_THRESHOLD = 0.7
CLUSTER_MIN_SIZE = 2
