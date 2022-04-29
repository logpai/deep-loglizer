import torch


class Optimizer:
    def __init__(self, parameter):
        self.optim = torch.optim.Adam(parameter, lr=2e-3, betas=(.9, .9),
                                      eps=1e-12)
        decay, decay_step = .75, 1000
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_last_lr()
