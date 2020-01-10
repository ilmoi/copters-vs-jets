from fastai.vision import *

path = Path('data/mil')
np.random.seed(42)
data = ImageDataBunch.from_csv(path, valid_pct=0.2, csv_labels='cleaned.csv',
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(6)
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn.export()
