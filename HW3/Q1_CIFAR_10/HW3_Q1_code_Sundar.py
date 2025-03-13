#!/usr/bin/env python
# coding: utf-8

# In[39]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import tqdm as tqdm


# In[40]:


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 128, shuffle = True)

images, labels = next(iter(trainloader))
classes = trainset.classes
plt.figure(figsize=(10,5))
plt.imshow(torchvision.utils.make_grid(images).permute(1,2,0) / 2 + 0.5)
plt.title(' '.join(classes[label] for label in labels[:4]));
plt.show()
test_losses = []


# In[41]:


class CIFAR10_NN(nn.Module):
    def __init__(self, activation_name, with_vg, with_bn):
        super(CIFAR10_NN, self).__init__()
        self.activation = activation_name
        self.with_vg = with_vg
        self.with_bn = with_bn
        self.fcLayer1 = nn.Linear(32*32*3, 512)
        self.fcLayer2 = nn.Linear(512, 256)
        self.fcLayer3 = nn.Linear(256, 128)
        
        # Fixing vanishing gradient
        if ( with_vg and with_bn ):
            self.fcbnLayer1 = nn.BatchNorm1d(128)
            self.fcbnLayer2 = nn.BatchNorm1d(64)
            self.fcbnLayer3 = nn.BatchNorm1d(32)
            self.fcbnLayer4 = nn.BatchNorm1d(16)
        
        self.fcLayer4 = nn.Linear(128, 64)
            
        self.fcLayer5 = nn.Linear(64, 32)
            
            
        self.fcLayer6 = nn.Linear(32, 16)
        self.fcLayervg1 = nn.Linear(16, 16)
        # Introduce Vanishing gradient by adding more layers
        if ( with_vg ):
            self.fcLayervg2 = nn.Linear(16, 16)
            self.fcLayervg3 = nn.Linear(16, 16)
            self.fcLayervg4 = nn.Linear(16, 16)
            self.fcLayervg5 = nn.Linear(16, 16)
            self.fcLayervg6 = nn.Linear(16, 16)
            self.fcLayervg7 = nn.Linear(16, 16)
            self.fcLayervg8 = nn.Linear(16, 16)
            self.fcLayervg9 = nn.Linear(16, 16)
        self.fcLayervg10 = nn.Linear(16, 16)
        # Fixing vanishing gradient
        if ( with_vg and with_bn ):
            self.initialize_weights()
        self.fcoLayer = nn.Linear(16, 10)
        if activation_name == "sigmoid": 
            self.activation = nn.Sigmoid()
        elif activation_name == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.fcLayer1(x))
        x = self.activation(self.fcLayer2(x))
        x = self.activation(self.fcLayer3(x))
        
        # Fixing vanishing gradient
        if ( self.with_vg and self.with_bn ):
            x = self.activation(self.fcbnLayer1(x))
        x = self.activation(self.fcLayer4(x))
        
        # Fixing vanishing gradient
        if ( self.with_vg and self.with_bn ):
            x = self.activation(self.fcbnLayer2(x))
        x = self.activation(self.fcLayer5(x))
        
        # Fixing vanishing gradient
        if ( self.with_vg and self.with_bn ):
            x = self.activation(self.fcbnLayer3(x))
        x = self.activation(self.fcLayer6(x))
        # Introduce Vanishing gradient by adding more layers
        if ( self.with_vg ):
            x = self.activation(self.fcLayervg1(x))
            x = self.activation(self.fcLayervg2(x))
            x = self.activation(self.fcLayervg3(x))
            x = self.activation(self.fcLayervg4(x))
            x = self.activation(self.fcLayervg5(x))
            x = self.activation(self.fcLayervg6(x))
            x = self.activation(self.fcLayervg7(x))
            x = self.activation(self.fcLayervg8(x))
            x = self.activation(self.fcLayervg9(x)) 
            x = self.activation(self.fcLayervg10(x))
        
        # Fixing vanishing gradient
        if ( self.with_vg and self.with_bn ):
            x = self.activation(self.fcbnLayer4(x))
        x = self.fcoLayer(x)
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight)  # He initialization for relu activation
                else:
                    nn.init.xavier_normal_(m.weight)  # Xavier initialization for sigmoid and tanh activations
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# In[42]:


def train_and_test(activation, with_vg, with_bn):
    device = torch.device("cpu")
    model = CIFAR10_NN(activation, with_vg, with_bn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    epochs = 10
    train_losses = []
    test_losses = []
    gradient_logs = {}
    for epoch in range(epochs):
        running_loss = 0.0
        train_acc=0.0
        with tqdm.tqdm(total=len(trainloader)) as pbar:
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss=criterion(outputs, labels)
                loss.backward()
                running_loss += loss.item()
                last_loss = loss.item()
                acc = torch.sum(torch.argmax(outputs, dim=1) == labels)
                train_acc += acc
                pbar.set_postfix(Loss='{0:.4f}'.format(loss.item()), Accuracy='{0:.4f}'.format(float(train_acc.item()/(images.size(0)*(batch_idx+1)))))
                pbar.update(1)
                if batch_idx == len(trainloader) - 1:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {last_loss:.4f}")
                    check_gradients(model, gradient_logs, epoch)
                
                optimizer.step()
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_losses.append(test_loss / len(testloader))
        average_loss = running_loss / len(trainloader)
        train_losses.append(average_loss)
        print(f"{epoch + 1} / {epochs}, activation = {activation}, loss = {average_loss: .4f}")
    return train_losses, test_losses, gradient_logs


# In[43]:


def check_gradients(model, gradient_logs, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if name not in gradient_logs:
                gradient_logs[name] = []
            if len(gradient_logs[name]) < epoch + 1:
                gradient_logs[name].append(grad_norm)
            else:
                gradient_logs[name][epoch] = grad_norm  # Ensure it logs per epoch


# In[44]:


activation_fns = ["sigmoid", "tanh", "relu"]
with_vgs = [False, True] # Simulate vanishing gradient - true and false
with_bns = [False, True] # Fix simulated vanishing gradient by applying weight initialization and batch normalization - true and false
results = {}
for activation in activation_fns:
    for vg in with_vgs:
        for bn in with_bns:
            if not vg and bn: continue # Not necessary to do without vanishing gradient and with fix
            key = f"{activation}_vg{vg}_bn{bn}"
            results[key] = (train_and_test(activation, vg, bn))


# In[45]:


# Plotting results for train and test loss
for name, result in results.items():
    print(result[0], result[1])
    plt.figure(figsize=(8, 5))
    plt.plot(result[0], label = 'Train Loss')
    plt.plot(result[1], label = 'Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title({name})
    plt.legend()
    plt.show()


# In[46]:


# Plotting gradients to show the effects of vanishing gradient and weight initialization
for name, result in results.items():
    _, _, gradient_logs = result  # Unpack the gradient logs
    plt.figure(figsize=(10, 6))
    for layer, norms in gradient_logs.items():
        plt.plot(norms, label=layer)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.yscale("log")
    plt.title(f"Gradient Norms - {name}")  # Add title with details
    # plt.legend()
    plt.show()

