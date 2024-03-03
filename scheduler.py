class TransformerLRScheduler:
    """
    Learning rate scheduler for Transformer models.

    Args:
        optimizer: The optimizer to update the learning rate for.
        d_model (int): Dimensionality of the model.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.d_model = d_model
        self._last_lr = []

    def step(self):
        """
        Updates the learning rate based on the current step.
        """
        self.step_num += 1
        calc_lr = (self.d_model ** -0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))

        values = [calc_lr]

        for data in zip(self.optimizer.param_groups, values):
            param_group, lr = data
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """
        Returns the last learning rate used for each parameter group.
        
        Returns:
            List[float]: The last learning rates.
        """
        return self._last_lr
