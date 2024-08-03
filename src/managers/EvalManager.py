from .TransformManager import TransformManager
from .MetricManager import MetricManager
from .DataManager import DataManager
from .ModelManager import ModelManager
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from .CustomMultiCropDataset import CustomMultiCropDataset

class EvalManager:
    @staticmethod
    def save_stats_to_file(file_path: str,
                           stats_dict: dict
                           ) -> None:
        with open(file_path, 'w') as file:
            json.dump(stats_dict, file, indent=4)
        
        print(f"[INFO] Saved statistics to {file_path}")

    @staticmethod
    def eval_single_scale(s_min,
                          s_max,
                          img_width, 
                          img_height, 
                          models_list, 
                          device,
                          batch_size,
                          num_workers,
                          stats_file_path
                        ):
        if s_min == s_max:
            test_scale = s_min
        else:
            test_scale = int((s_min + s_max) / 2)

        print(f"\n[INFO] Evaluating models at single test scale,Q: {test_scale}\n")

        stats_dict = {}
        test_transform = TransformManager.get_test_transform_single_scale(test_scale,img_width,img_height)
        test_dl = DataManager.get_test_dl(test_transform, 
                                                    batch_size,
                                                    num_workers
                                                )
        for model in models_list:
            print(f"[INFO] Performing eval for model {model.name}")
            preds, labels, probs = ModelManager.predict(model=model, 
                                                        dataloader=test_dl,
                                                        device=device
                                                        )
            top_1_err = MetricManager.get_top_1_err(preds, labels)
            top_5_err = MetricManager.get_top_5_err(probs, labels)
            acc = MetricManager.get_accuracy(preds, labels)

            stats = {
                "s_min": s_min,
                "s_max": s_max,
                "test_scale": test_scale,
                "top_1_err": top_1_err,
                "top_5_err": top_5_err,
                "acc": acc
            }
            
            stats_dict[model.name] = stats
            
            print(f"[INFO] Top 1 Err: {top_1_err}")
            print(f"[INFO] Top 5 Err: {top_5_err}")
            print(f"[INFO] Accuracy: {acc}\n")

        print("=" * 50)

        EvalManager.save_stats_to_file(stats_file_path, stats_dict)
        return stats_dict

    @staticmethod
    def eval_multi_scale(s_min, 
                         s_max, 
                         img_width, 
                         img_height, 
                         models_list, 
                         device, 
                         batch_size, 
                         num_workers,
                         stats_file_path
                        ):
        if s_min == s_max:
            test_scales = [s_min - 10, s_min , s_min + 10]
        else:
            test_scales = [s_min, int(0.5 * (s_min + s_max)), s_max]
        
        stats_dict = {}
        print(f"\n[INFO] Evaluating models at mulitple test scales,Q: {test_scales}\n")
        
        for model in models_list:
            print(f"[INFO] Performing eval for model {model.name}")
            test_transforms = [TransformManager.get_test_transform_single_scale(scale, img_width, img_height) for scale in test_scales]
            test_dls = [DataManager.get_test_dl(transform, batch_size, num_workers) for transform in test_transforms]
            probs_lis = []
            for dl in test_dls:
                preds, labels, probs = ModelManager.predict(model=model, 
                                                            dataloader=dl,
                                                            device=device
                                                                )
                probs_lis.append(probs)
        
            avg_probs = torch.mean(torch.stack(probs_lis), dim=0)
            final_preds = torch.argmax(avg_probs, dim=1)
            top_1_err = MetricManager.get_top_1_err(final_preds, labels)
            top_5_err = MetricManager.get_top_5_err(avg_probs, labels)
            acc = MetricManager.get_accuracy(final_preds, labels)
            stats = {
                "s_min": s_min,
                "s_max": s_max,
                "test_scales": test_scales,
                "top_1_err": top_1_err,
                "top_5_err": top_5_err,
                 "acc": acc
            }
            stats_dict[model.name] = stats 
        
            print(f"[INFO] Top 1 Err: {top_1_err}")
            print(f"[INFO] Top 5 Err: {top_5_err}")
            print(f"[INFO] Accuracy: {acc}\n")
        
        print("=" * 50)
        EvalManager.save_stats_to_file(stats_file_path, stats_dict)
        return stats_dict

    @staticmethod
    def eval_multi_crop(s_min,
                        s_max,
                        img_width, 
                        img_height, 
                        test_scales,
                        models_lis,
                        device,
                        batch_size,
                        num_workers,
                        stats_file_path
                        ):
        dataset = load_dataset("zh-plus/tiny-imagenet")
        test_ds = dataset["valid"]
        test_ds = test_ds.select(range(80))
        custom_test_ds = CustomMultiCropDataset(test_ds, test_scales, img_width)
        print(test_scales)
        test_dl= DataLoader(custom_test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        stats_dict = dict()
        for model in models_lis:
            print(f"[INFO] Evaluating model: {model.name}")
            model.eval()
            all_preds = []
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for crops, labels in test_dl:
                    labels = labels.to(device)
                    crops = crops.to(device)  # Move crops to device

                    # Forward pass through the model
                    outputs = model(crops)

                    # Convert outputs to probabilities
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu())

                    # Get the predictions (max over the probabilities)
                    _, preds = probs.max(1)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
            
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            all_probs = torch.cat(all_probs)

            top_1_err = MetricManager.get_top_1_err(all_preds, all_labels)
            top_5_err = MetricManager.get_top_5_err(all_probs, all_labels)
            acc = MetricManager.get_accuracy(all_preds, all_labels)

            stats = {
                "s_min": s_min,
                "s_max": s_max,
                "test_scales": test_scales,
                "top_1_err": top_1_err,
                "top_5_err": top_5_err,
                 "acc": acc
            }

            stats_dict[model.name] = stats


            print(f"[INFO] Top 1 Err: {top_1_err}")
            print(f"[INFO] Top 5 Err: {top_5_err}")
            print(f"[INFO] Accuracy: {acc}\n")
        
        print("=" * 50)
        EvalManager.save_stats_to_file(stats_file_path, stats_dict)

        return stats_dict
    