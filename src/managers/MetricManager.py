import torch

class MetricManager:
    @staticmethod
    def get_top_1_err(preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculate the Top-1 Error Rate.
        
        Args:
        preds (torch.Tensor): Tensor of predicted class indices.
        labels (torch.Tensor): Tensor of true labels.
        
        Returns:
        float: Top-1 error rate.
        """
        top_1_err = torch.mean((preds != labels).float()).item()
        return top_1_err

    @staticmethod
    def get_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculate the accuracy.
        
        Args:
        preds (torch.Tensor): Tensor of predicted class indices.
        labels (torch.Tensor): Tensor of true labels.
        
        Returns:
        float: Accuracy.
        """
        accuracy = torch.mean((preds == labels).float()).item()
        return accuracy

    @staticmethod
    def get_top_5_err(probs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculate the Top-5 Error Rate.
        
        Args:
        probs (torch.Tensor): Tensor of probability distributions.
        labels (torch.Tensor): Tensor of true labels.
        
        Returns:
        float: Top-5 error rate.
        """
        _, top_5_preds = torch.topk(probs, 5, dim=1, largest=True, sorted=True)
        labels = labels.view(-1, 1).expand_as(top_5_preds)
        top_5_err = torch.mean((top_5_preds != labels).all(dim=1).float()).item()
        return top_5_err


