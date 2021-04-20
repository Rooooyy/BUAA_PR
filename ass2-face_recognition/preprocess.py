import os

IMG_PATH = "./data/Face Recognition Data/grimace/"

for folder in os.listdir(IMG_PATH):
    path_folder = os.path.join(IMG_PATH, folder)
    for file_name in os.listdir(path_folder):
        file_name_new = file_name.split('.')[0].split('_')[0] + '.' \
                        + file_name.split('.')[1] + '.' + file_name.split('.')[2]
        src = os.path.join(path_folder, file_name)
        dst = os.path.join(path_folder, file_name_new)
        os.rename(src, dst)


