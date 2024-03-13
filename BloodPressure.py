import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch import optim as optim
from tqdm import tqdm

plt.style.use('seaborn')

data = pd.read_csv('Blood-Pressure.csv')
data.info()
data.describe()
data.head()

glucose_mean = 120.894531
glucose_std = 31.972618

bp_mean = 69.105469
bp_std = 19.355807

glucose = ((torch.tensor(data['Glucose'], dtype=torch.float) - glucose_mean) / glucose_std)[:384]
bp = ((torch.tensor(data['BloodPressure'], dtype=torch.float) - bp_mean) / bp_mean)[:384]

test_glucose = ((torch.tensor(data['Glucose'], dtype=torch.float) - glucose_mean) / glucose_std)[-384:]
test_bp = ((torch.tensor(data['BloodPressure'], dtype=torch.float) - bp_mean) / bp_mean )[-384:]

plt.figure(figsize=(18, 4.8))

plt.subplot(1, 3, 1)
plt.scatter(glucose, bp)
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.title('Glucose vs. Blood Pressure Scatter Plot')

plt.subplot(1, 3, 2)
plt.scatter(glucose / glucose.max(), bp * glucose, c='red')
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.title('Normalized Glucose vs. Glucose * Blood Pressure Scatter Plot')

plt.subplot(1, 3, 3)
plt.scatter(glucose / glucose.max(), bp + 0.9 * glucose, c='orange')
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.title('Normalized Glucose vs. Blood Pressure + 0.9 * Glucose Scatter Plot')

weights = torch.randn((len(glucose)), dtype=torch.float, requires_grad=True)
bias = torch.randn(1, dtype=torch.float, requires_grad=True)

def linear(x, w, b):
    res = x * w + b
    return res

cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD([weights, bias], lr=0.1)
print(glucose.shape, weights.shape)

epochs = 15000
accumu_loss = []

for e in tqdm(range(epochs)):
    pred = linear(glucose, weights, bias)
    loss = cost(pred, bp)
    loss.backward()
    optimizer.step()
    accumu_loss.append(loss.detach().numpy())
    optimizer.zero_grad()
print(f'loss:{loss}')

plt.plot(accumu_loss, 'orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs Plot')
plt.show()

def plot_fit(x, w, b):
    w1 = w.detach().numpy().mean()
    b1 = b.detach().numpy()
    y1 = x * w1 + b1
    plt.plot(x, y1, 'orange')
    print('Bias: ', b1)

plt.scatter(glucose, bp, c='purple')
plot_fit(np.array([[-4], [2.5]]), weights, bias)
plt.title('Glucose vs. Blood Pressure with Fitted Line')
plt.show()

sns.regplot(x=glucose.numpy(), y=bp.numpy(), color='blue')
plt.title('Glucose vs. Blood Pressure Regression Plot')
plt.show()

plt.scatter(test_glucose, test_bp, c='purple')
plot_fit(np.array([[-4], [3]]), weights, bias)
plt.title('Test Data: Glucose vs. Blood Pressure with Fitted Line')
plt.show()

with torch.no_grad():
    pred = linear(test_glucose, weights, bias)
    loss = cost(pred, test_bp).item()
print('loss: ', loss)