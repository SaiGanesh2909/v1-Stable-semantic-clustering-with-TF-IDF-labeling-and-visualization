import os
import shutil


def move_file_to_cluster(root, file_path, folder_name):

    if not os.path.exists(file_path):
        return

    cluster_folder = os.path.join(root, folder_name)
    os.makedirs(cluster_folder, exist_ok=True)

    filename = os.path.basename(file_path)
    destination = os.path.join(cluster_folder, filename)

    if os.path.abspath(file_path) == os.path.abspath(destination):
        return

    if os.path.exists(destination):
        return

    try:
        shutil.move(file_path, destination)
    except:
        pass
