import multiprocessing
import torch
import src.managers as managers
import src.model as model
import torch.nn as nn

if __name__ == "__main__":
    BATCH_SIZE = 256
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    DROPOUT_PROBA = 0.5
    NUM_WORKERS = multiprocessing.cpu_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIM = 32
    NUM_EPOCHS = 1
    MODELS_FOLDER = "/content/drive/MyDrive/Computer-Vision/papers/VGG/models"
    LEARNING_RATE = 1e-2

    for scale in [44,54]:
        print("[INFO] Training Models for training scale, S: {scale}")
        train_transform = managers.TransformManager.get_train_transform(s_min=scale, s_max=scale, img_height=32, img_width=32)                                    )
        train_dl = managers.DataManager.get_train_dl(train_transform, BATCH_SIZE, NUM_WORKERS)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.)
        optimizer = 
        for name in model.Arch.ARCH.keys():
            print(f"Model: {name}")
            model = model.VGG(name)
            managers.ModelManager.train(model,)




