NEPTUNE_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTI1ODlhMTYtM2I5NC00MjdmLWE5YWYtZjc0YWI5OGNhOGI5In0='

TRAIN_FOLDER = '/home/wsz_dl1/dataset/cars_train/'
PREPROCESS_FOLDER = '/home/wsz_dl1/dataset/cars_preprocess/'
#TRAIN_FOLDER = 'D:\\cars_dataset\\30084_38348_bundle_archive\\cars_train\\cars_train'

#LABEL_FILE_TRAIN = 'D:\\cars_dataset\\30084_38348_bundle_archive\\cars_train_annos.mat'
LABEL_FILE_TRAIN = '/home/wsz_dl1/dataset/devkit/cars_train_annos.mat'

NUM_CLASSES = 197

BATCH_SIZE = 2048
TRAIN_PERC = 0.9
MAX_EPOCH = 1000

SCALE_CHANNELS = {
    'scale_1': 16,
    'scale_2': 32,
    'scale_3': 64
}
