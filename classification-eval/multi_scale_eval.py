import torch
import src.managers as managers
import src.model_components as model_components
import torch.nn as nn
import os
from SingleScaleEvalStats import SingleScaleEvalStats
from MultiScaleEvalStats import MultiScaleEvalStats
import json

def save_eval_stats_to_file(stats: list, file_path: str) -> None:
    """
    Save evaluation statistics to a file.

    Args:
    stats (list): List of SingleScaleEvalStats objects.
    file_path (str): Path to the file where the stats will be saved.
    """
    output = {}
    for s in stats:
        output[s.model_name] = {
            's_min': s.train_min_scale,
            's_max': s.train_max_scale,
            'test_scales': s.test_scales,
            'top_1_err': s.top_1_err,
            'top_5_err': s.top_5_err,
            'acc': s.acc
        }
    
    with open(file_path, 'w') as file:
        json.dump(output, file, indent=4)

    print(f"[INFO] Saved statistics to {file_path}")

def eval(s_min, s_max, img_width, img_height, models_list, device, batch_size, num_workers):
    stats_lis = []
    # if the training scale is single scaled
    if s_min == s_max:
        test_scales = [s_min - 10, s_min , s_min + 10]
    else:
        test_scales = [s_min, int(0.5 * (s_min + s_max)), s_max]
    
    for model in models_list:
        print(f"[INFO] Performing eval for model {model.name}")
        test_transforms = [managers.TransformManager.get_test_transform_single_scale(scale, img_width, img_height) for scale in test_scales]
        test_dls = [managers.DataManager.get_test_dl(transform, batch_size, num_workers) for transform in test_transforms]
        probs_lis = []
        for dl in test_dls:
            preds, labels, probs = managers.ModelManager.predict(model=model, 
                                                                 dataloader=dl,
                                                                 device=device
                                                                )
            probs_lis.append(probs)
        
        avg_probs = torch.mean(torch.stack(probs_lis), dim=0)
        final_preds = torch.argmax(avg_probs, dim=1)
        top_1_err = managers.MetricManager.get_top_1_err(final_preds, labels)
        top_5_err = managers.MetricManager.get_top_5_err(avg_probs, labels)
        acc = managers.MetricManager.get_accuracy(final_preds, labels)
        stats_lis.append(MultiScaleEvalStats(model_name=model.name,
                                train_min_scale=s_min,
                                train_max_scale=s_max,
                                test_scales=test_scales,
                                top_1_err=top_1_err,
                                top_5_err=top_5_err,
                                acc=acc
                                )
                        )
        print(f"[INFO] Top 1 Err: {top_1_err}")
        print(f"[INFO] Top 5 Err: {top_5_err}")
        print(f"[INFO] Accuracy: {acc}\n")
    
    print("=" * 50)
    return stats_lis


if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_WORKERS = 0
    WEIGHTS_FOLDER = "weights-classification"
    MULTISCALE_WEIGHTS = f"{WEIGHTS_FOLDER}/multiscale"
    SCALE_44_WEIGHTS = f"{WEIGHTS_FOLDER}/scale_44"
    SCALE_54_WEIGHTS = f"{WEIGHTS_FOLDER}/scale_54"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIM = 32
    RESULTS_FOLDER = "classification-eval/results_multi_scale"

    train_scale_44_models = managers.ModelManager.load_models(SCALE_44_WEIGHTS)
    # train_scale_54_models = managers.ModelManager.load_models(SCALE_54_WEIGHTS)
    # multiscale_models = managers.ModelManager.load_models(MULTISCALE_WEIGHTS)
    train_scale_44_stats = eval(s_min=44, 
                                s_max=44, 
                                img_width=IMG_DIM, 
                                img_height=IMG_DIM, 
                                models_list=train_scale_44_models, 
                                device=DEVICE,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS
                                )
    save_eval_stats_to_file(train_scale_44_stats,f"{RESULTS_FOLDER}/train_scale_54_stats.json")


