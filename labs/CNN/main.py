import numpy as np
import torch
import torch.nn as nn
from torchviz import make_dot

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

DATA_PATH = 'E:\\acady\\learning\\ml\\labs\\CNN\\datasets'
MODEL_PATH = 'E:\\acady\\learning\\ml\\labs\\CNN\\pytorch_models\\'

trans = transforms.ToTensor()

# MNIST
# train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
# test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# MNIST fashion
train_dataset = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=False, transform=trans)

# targets = train_dataset.targets

batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


class ConvNet(nn.Module):
    def __init__(self, layer1_size=32, layer2_size=64):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, layer1_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_size, layer2_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * layer2_size, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def learnModel(model, train_loader, num_epochs, device):
    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    steps_cnt = len(train_loader)
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device=device), labels.to(device=device)

            # plt.imshow(images[0].cpu().permute(1, 2, 0))
            # plt.show()

            outputs = model(images)
            # make_dot(outputs).render("test", format="png")
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, steps_cnt, loss.item(),
                              (correct / total) * 100))

    return model, acc_list


# %%

# labels = [i for i in range(10)]
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# %%

def testModel(model, device, calc_stats=False):
    model.to(device=device)
    model.eval()

    softmax = nn.Softmax(dim=0)
    best_prob = [(0, None) for i in range(len(labels))]
    confusion_matrix = np.zeros(shape=[len(labels), len(labels)])

    with torch.no_grad():
        correct = 0
        total = 0
        for images, targets in test_loader:
            images, targets = images.to(device=device), targets.to(device=device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if (calc_stats):
                for output, target, prediction, image in zip(outputs, targets, predicted, images):
                    probs = softmax(output)
                    for i, prob in enumerate(probs):
                        if prob > best_prob[i][0]:
                            best_prob[i] = (prob.item(), image)

                    confusion_matrix[prediction][target] += 1

        test_accuracy = correct / total
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(test_accuracy * 100))

        return test_accuracy, confusion_matrix, best_prob


# %%

# for i, (images, labels) in enumerate(train_loader):
#     plt.figure()
#     plt.imshow(images[0].cpu().permute(1, 2, 0))
#     plt.colorbar()
#     plt.show()
#     break

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(layer1_size=64, layer2_size=65)

# model.to(device=device)
# model.load_state_dict(torch.load(MODEL_PATH + 'fashion.ckpt', map_location=device))

model, _ = learnModel(model, train_loader, num_epochs=5, device=device)
_, confusion_matrix, best_prob = testModel(model, device, calc_stats=True)

torch.save(model.state_dict(), MODEL_PATH + 'fashion.ckpt')

# %%

plt.figure(figsize=(10, 4))
for i in range(len(best_prob)):
    plt.subplot(2, 5, i + 1)
    plt.title("{} : {:.5f}".format(labels[i], best_prob[i][0]))
    plt.imshow(best_prob[i][1].cpu().permute(1, 2, 0))
plt.tight_layout()
plt.show()


# %%

def findOptimal(layer1_range, layer2_range, epochs):
    best_test_accuracy, best_model_params = 0, (-1, -1)
    for layer1_size in layer1_range:
        for layer2_size in layer2_range:
            model = ConvNet(layer1_size, layer2_size)

            model, _ = learnModel(model, train_loader, num_epochs=epochs, device=device)
            accuracy, _, _ = testModel(model, device)

            best_test_accuracy, best_model_params = max((best_test_accuracy, best_model_params),
                                                        (accuracy, (layer1_size, layer2_size)))

    return best_test_accuracy, best_model_params


layer_range = [1, 2, 4, 8, 16, 32, 64, 128]
best_acc, best_model = findOptimal(layer_range, layer_range, epochs=5)
