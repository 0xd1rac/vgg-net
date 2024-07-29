from typing import List

class MultiScaleEvalStats:
    def __init__(self,
                 model_name: str,
                 train_min_scale: float,
                 train_max_scale: float,
                 test_scales: List[int],
                 top_1_err: float,
                 top_5_err: float,
                 acc: float
                 ):
        self.model_name = model_name
        self.train_min_scale = train_min_scale
        self.train_max_scale = train_max_scale
        self.test_scales = test_scales
        self.top_1_err = top_1_err
        self.top_5_err = top_5_err
        self.acc = acc

    def display_stats(self):
        print(f"Model Name: {self.model_name}")
        print(f"Training Min Scale: {self.train_min_scale}")
        print(f"Training Max Scale: {self.train_max_scale}")
        print(f"Test scales: {self.test_scales}")
        print(f"Top-1 Error Rate: {self.top_1_err:.4f}")
        print(f"Top-5 Error Rate: {self.top_5_err:.4f}")
        print(f"Accuracy: {self.acc:.4f}")
