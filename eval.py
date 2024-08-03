import src.managers as managers
import argparse
import json

def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to load and use a JSON config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON config file")
    parser.add_argument('--test-mode', type=str, choices=['multi-scale', 'multi-crop', 'single-scale'], required=True, help="Test mode options: multi-scale, multi-crop, single-scale")
    args = parser.parse_args()
    config = load_config(args.config)
    test_mode = args.test_mode
    print(f"Test Mode: {test_mode}")
    models = managers.ModelManager.load_models(config['weights_folder_path'])
    
    if test_mode == 'multi-scale' or test_mode == 'single-scale':
        print("multi-scale")
        eval_args = {
            's_min': config['s_min'],
            's_max': config['s_max'],
            'img_width': config['img_width'],
            'img_height': config['img_height'],
            'models_list': models,  # Assuming 'models' is defined elsewhere in your code
            'device': config['device'],
            'batch_size': config['batch_size'],
            'num_workers': config['num_workers'],
            'stats_file_path': config['eval_results_file']
        }
        if test_mode == 'single-scale':
            managers.EvalManager.eval_single_scale(**eval_args)
        
        else:
            managers.EvalManager.eval_multi_scale(**eval_args)

    else:
        print("MULTI-CROP")
        managers.EvalManager.eval_multi_crop(
            s_min=config['s_min'],
            s_max=config['s_max'],
            img_width = config['img_width'],
            img_height=config['img_height'],
            test_scales = config['test_scales'],
            models_lis = models,
            device= config['device'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            stats_file_path= config['eval_results_file']
        )


