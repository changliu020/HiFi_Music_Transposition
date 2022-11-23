from model import *
from trainer import Trainer
import torch

fe = FrontEnd()
d = D()
q = Q()
g = G()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in [fe, d, q, g]:
    i.to(device)
    i.apply(weights_init)

trainer = Trainer(g, fe, d, q, device)
trainer.train()
