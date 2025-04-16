import kagglehub
import os

save_path = os.getcwd()

path = kagglehub.dataset_download(
    handle="tabassum18/handwritten-marathi-character-augmented-dataset", path=save_path
)

print("Path to dataset files:", path)
