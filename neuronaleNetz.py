import torch
import torch.nn as nn
import torch.nn.functional as F

class MeinNetz(nn.Module):
    # Konstruktor, definiert was für Variables das Neuronale Netz später hat,
    # Jedes Layer in dem neuronalen Netz, Converlution Layer, Linear Layer ist eine eigene Variable
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    # Forward Path rechnet den Input durch, bekommt input rein
    def forward(self, x):
        # relu: eine Aktivierungsfunktion auf unserer Schicht
        x = F.relu(self.lin1(x))
        x = self.lin2(x)    # Das letzte sollte immer ohne Aktivierungsfunktion gemacht werden, damit wir Output bekommen können
        return x

    # Eine Funktion damit nn.Module ausgeben kann
    def num_flat_features(self, x):
        # Batch Dimenstion erstmal auslassen
        size = x.size()[1:]
        num = 1         # Ausrechnen wie viele Features wir haben
        for i in size:
            # x ist tensor, wir sagen z.b. 5*3 features
            num *= i
        return num

netz = MeinNetz()
print(netz)