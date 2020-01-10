from fastai.vision import *

classes = ['helicopter','fighter_jet']
for c in classes:
    folder = c
    file = f"{c}.csv"
    path = Path('data/mil')
    dest = path/folder
    dest.mkdir(parents=True, exist_ok=True)
    download_images(path/file, dest, max_pics=200)
    verify_images(path/c, delete=True, max_size=500)
