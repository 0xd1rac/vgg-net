import torch
import src.managers as managers

if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_WORKERS = 0
    WEIGHTS_FOLDER = "weights-classification"
    MULTISCALE_WEIGHTS = f"{WEIGHTS_FOLDER}/multiscale"
    SCALE_44_WEIGHTS = f"{WEIGHTS_FOLDER}/scale_44"
    SCALE_54_WEIGHTS = f"{WEIGHTS_FOLDER}/scale_54"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIM = 32
    RESULTS_FOLDER = "classification-eval-results/single_scale"

    train_scale_44_models = managers.ModelManager.load_models(SCALE_44_WEIGHTS)

    managers.EvalManager.eval_single_scale(
        s_min=44,
        s_max=44,
        img_width=IMG_DIM,
        img_height=IMG_DIM,
        models_list=train_scale_44_models,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        stats_file_path=f"{RESULTS_FOLDER}/train_scale_44_models.json"
    )
    
    train_scale_54_models = managers.ModelManager.load_models(SCALE_54_WEIGHTS)
    managers.EvalManager.eval_single_scale(
        s_min=54,
        s_max=54,
        img_width=IMG_DIM,
        img_height=IMG_DIM,
        models_list=train_scale_54_models,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        stats_file_path=f"{RESULTS_FOLDER}/train_scale_54_models.json"
    )

    multiscale_models = managers.ModelManager.load_models(MULTISCALE_WEIGHTS)
    managers.EvalManager.eval_single_scale(
        s_min=44,
        s_max=54,
        img_width=IMG_DIM,
        img_height=IMG_DIM,
        models_list=train_scale_54_models,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        stats_file_path=f"{RESULTS_FOLDER}/train_multiscale_models.json"
    )
 