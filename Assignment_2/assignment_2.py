
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 1 exr
path = '../dataSet_CIFAR10'
train_dataset = torchvision.datasets.CIFAR10(path, train=True, download=True) #, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(path, train=False, download=True) #, transform=transforms.ToTensor()) 


rand_values = []  
for i in range(5):
    rand_values.append(random.randint(4, 50))

for i in range(len(rand_values)):
    img, label = train_dataset[rand_values[i]]
    print(f'img: {img}')
    print(f'label: {label}')
    plt.imshow(img)
    plt.show()


torch.manual_seed(0)
np.random.seed(0)

## 2 exr
my_tranforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))])

path = '../dataSet_CIFAR10'
train_dataset = torchvision.datasets.CIFAR10(path, train=True, transform=my_tranforms)
test_dataset = torchvision.datasets.CIFAR10(path, train=False, transform=my_tranforms)



for i in range(len(rand_values)):
    img, label = train_dataset[rand_values[i]]
    print(f'img: {img}')
    print(f'label: {label}')
    break
  # plt.imshow(img)
  # plt.show()

print(f'Max value of a random picture is: {img.max()}')
print(f'Min value of a random picture is: {img.min()}')


## 3 exr

means = []
std = []
for i in range(len(train_dataset)):
    means.append(train_dataset[i][0].view(3,-1).mean(dim=1))
    std.append(train_dataset[i][0].view(3, -1).std(dim=1))


first_channel_mean = []
second_channel_mean = []
third_channel_mean = []

first_channel_std = []
second_channel_std = []
third_channel_std = []

for i in range(len(means)):
    first_channel_mean.append(means[i][0])
    second_channel_mean.append(means[i][1])
    third_channel_mean.append(means[i][2])

    first_channel_std.append(std[i][0])
    second_channel_std.append(std[i][1])
    third_channel_std.append(std[i][2])

print(f'Mean across the First channel: {torch.mean(torch.stack(first_channel_mean))}')
print(f'Mean across the Second channel: {torch.mean(torch.stack(second_channel_mean))}')
print(f'Mean across the Third channel: {torch.mean(torch.stack(third_channel_mean))}')
print('-----------------------')
print(f'Std across the First channel: {torch.mean(torch.stack(first_channel_std))}')
print(f'Std across the Second channel: {torch.mean(torch.stack(second_channel_std))}')
print(f'Std across the Third channel: {torch.mean(torch.stack(third_channel_std))}')


## 4 exr
batch_size = 32

idx = np.arange(len(train_dataset))

# Use last 5000 images for validation
val_indices = idx[50000-1000:]
train_indices= idx[:-1000]


print(len(val_indices))
print(len(train_indices))


from numpy.random.mtrand import shuffle

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)

valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          sampler=valid_sampler, num_workers=2)


## 1.2 model

## 1 exr


batch_size = 32
num_classes = 10
epochs = 20
learning_rate = 0.001
momentum = 0.9

class Conv_network(nn.Module):
    def __init__(self):
        super(Conv_network, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3) # Convolutional layer 1: 32 filters, 3 × 3
 
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3) # Convolutional layer 2: 32 filters, 3 × 3
      
        self.pool_1 = nn.MaxPool2d(2, 2) # Max-pooling layer 1: 2 × 2 windows
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3) # Convolutional layer 3: 64 filters, 3 × 3
     
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # Convolutional layer 4: 64 filters, 3 × 3
        self.pool_2 = nn.MaxPool2d(2, 2) # Max-pooling layer 2: 2 × 2 windows

        self.fc1 = nn.Linear(64*5*5, 512) # Fully connected layer 1: 512 units
        self.fc2 = nn.Linear(512, 10) # Output layer with num_classes


    def forward(self, x):

        x = F.relu(self.conv_1(x))      # conv_1 
        
        x = self.pool_1(F.relu(self.conv_2(x))) # conv_2 + pool_1
        
        x = F.relu(self.conv_3(x)) # conv_3
        
        x = self.pool_2(F.relu(self.conv_4(x))) # conv_4 + pool_2

        x = nn.Flatten()(x)
        
        x = F.relu(self.fc1(x)) # fully connected layer
        x = self.fc2(x) # fully connected layer + output layer

        return x



## 1.3 Training

model = Conv_network().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True



# Training
train_loss_steps = []
train_acc_steps = []

train_loss_epochs = []
train_acc_epochs = []

valid_acc = []
valid_loss = []
for epoch in range(1, epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # put the model to train mode
        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = model(images)  # shape (B, 10).
        loss = loss_fn(outputs, labels)
        #train_loss.append(loss)
        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss = loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 500 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  #f'train_loss: {running_loss / run_step :.3f}, '
                  f'train_loss: {round(running_loss, 3)}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            
            train_loss_steps.append(round(running_loss, 3)) 
            train_acc_steps.append(round(running_correct / running_total, 3))
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
            
    train_loss_epochs.append(*train_loss_steps[-1:]) # add final value from each epoch 
    train_acc_epochs.append(*train_acc_steps[-1:])

    # validation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
#             images = images.view(-1, 32*32)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            validation_loss = loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        valid_loss.append(round(validation_loss, 3))
        valid_acc.append(round(correct / total, 3))
        #valid_loss.append(1 - round(correct/total,3))
    print(f'Validation accuracy: {100 * correct / total: .1f} %')
    
print('Finished Training')




iters = np.arange(1,20,1) 
print(f'len(train_loss_epochs) - {len(train_loss_epochs)}')
print(f'len(train_acc_epochs) - {len(train_acc_epochs)}')
print(f'len(valid_acc) - {len(valid_acc)}')
print(f'len(valid_loss) - {len(valid_loss)}')


## plot loss
fig , ax = plt.subplots(figsize=(10, 8))
ax.plot(iters, train_loss_epochs, color='blue', label='Train')
ax.plot(iters, valid_loss, color='red', label='Valid')

ax.legend(loc='best')

ax.set_xlabel ("epochs", fontsize =16)
ax.set_ylabel ("CrossEntropyLoss", fontsize =16)




## plot accuracy
fig , ax = plt.subplots(figsize=(10, 8))
ax.plot(iters, train_acc_epochs, color='blue', label='Train')
ax.plot(iters, valid_acc, color='red', label='Valid')

ax.legend(loc='best')

ax.set_xlabel ("epochs", fontsize =16)
ax.set_ylabel ("Accuracy", fontsize =16)




## Dropout (p = 0.2) ---------------------------------

batch_size = 32
num_classes = 10
epochs = 30
learning_rate = 0.001
momentum = 0.9

class Conv_network(nn.Module):
    def __init__(self):
        super(Conv_network, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3) # Convolutional layer 1: 32 filters, 3 × 3
 
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3) # Convolutional layer 2: 32 filters, 3 × 3
      
        self.pool_1 = nn.MaxPool2d(2, 2) # Max-pooling layer 1: 2 × 2 windows
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3) # Convolutional layer 3: 64 filters, 3 × 3
     
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # Convolutional layer 4: 64 filters, 3 × 3
        self.pool_2 = nn.MaxPool2d(2, 2) # Max-pooling layer 2: 2 × 2 windows

        self.fc1 = nn.Linear(1600, 512) # Fully connected layer 1: 512 units
        self.fc2 = nn.Linear(512, 10) # Output layer with num_classes
        
        self.dropout = nn.Dropout(0.2)
        #self.activation = nn.ReLU(inplace=True) # linear activation functions (ReLUs)


    def forward(self, x):

        x = F.relu(self.conv_1(x))      # conv_1 
        
        x = self.pool_1(F.relu(self.conv_2(x))) # conv_2 + pool_1
        
        x = self.dropout(x)
        
        x = F.relu(self.conv_3(x)) # conv_3
        
        x = self.pool_2(F.relu(self.conv_4(x))) # conv_4 + pool_2
        
        x = self.dropout(x)
        
        x = nn.Flatten()(x)
        
        x = F.relu(self.fc1(x)) # fully connected layer
        x = self.fc2(x) # fully connected layer + output layer
        return x


model = Conv_network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Training
train_loss_steps = []
train_acc_steps = []

train_loss_epochs = []
train_acc_epochs = []

valid_acc = []
valid_loss = []
for epoch in range(1, epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # put the model to train mode
        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = model(images)  # shape (B, 10).
        loss = loss_fn(outputs, labels)
        #train_loss.append(loss)
        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss = loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 500 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  #f'train_loss: {running_loss / run_step :.3f}, '
                  f'train_loss: {round(running_loss, 3)}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            
            train_loss_steps.append(round(running_loss, 3)) 
            train_acc_steps.append(round(running_correct / running_total, 3))
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
            
    train_loss_epochs.append(*train_loss_steps[-1:]) # add final value from each epoch 
    train_acc_epochs.append(*train_acc_steps[-1:])

    # validation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
#             images = images.view(-1, 32*32)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            validation_loss = loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        valid_loss.append(round(validation_loss, 3))
        valid_acc.append(round(correct / total, 3))
        #valid_loss.append(1 - round(correct/total,3))
    print(f'Validation accuracy: {100 * correct / total: .1f} %')
    
print('Finished Training')


iters = np.arange(1,30,1) 
print(f'len(train_loss_epochs) - {len(train_loss_epochs)}')
print(f'len(train_acc_epochs) - {len(train_acc_epochs)}')
print(f'len(valid_acc) - {len(valid_acc)}')
print(f'len(valid_loss) - {len(valid_loss)}')




fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Dropout p = 0.2")
ax.plot(iters, train_loss_epochs, color='blue', label='Train')
ax.plot(iters, valid_loss, color='red', label='Valid')
ax.legend(loc='best')
ax.set_xlabel ("epochs", fontsize =16)
ax.set_ylabel ("CrossEntropyLoss", fontsize =16)



fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Dropout p = 0.2")
ax.plot(iters, train_acc_epochs, color='blue', label='Train')
ax.plot(iters, valid_acc, color='red', label='Valid')
ax.legend(loc='best')
ax.set_xlabel ("epochs", fontsize =16)
ax.set_ylabel ("Accuracy", fontsize =16)



correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        validation_loss = loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
    #valid_loss.append(round(validation_loss, 3))
    #valid_acc.append(round(correct / total, 3))
print(f'Test accuracy: {100 * correct / total: .1f} %')



## here i will visialize random pictures from ("Report the test set accuracy of the best model you obtained above")

rand_values = []  
for i in range(5):
    rand_values.append(random.randint(4, 50))
    
for i in range(len(rand_values)):
    test_loader_1 = torch.utils.data.DataLoader(dataset=test_dataset[rand_values[i]], batch_size=1, shuffle=False)
    images, labels = test_loader_1
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    
    _, predicted = outputs.max(1)
    print(outputs)
    #correct += (predicted == labels).sum().item()
    print(f'Predict_class:   {predicted.item()}')
    #print(len(predicted))
    print(f'Real_class:      {labels.item()}')
    img, label = test_dataset[rand_values[i]]
    plt.imshow(img.permute(1, 2, 0))
    #plt.imshow(img)
    plt.show()
    print('-------')



## Dropout (p = 0.5) ------------------------

class Conv_network(nn.Module):
    def __init__(self):
        super(Conv_network, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3) # Convolutional layer 1: 32 filters, 3 × 3
 
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3) # Convolutional layer 2: 32 filters, 3 × 3
      
        self.pool_1 = nn.MaxPool2d(2, 2) # Max-pooling layer 1: 2 × 2 windows
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3) # Convolutional layer 3: 64 filters, 3 × 3
     
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # Convolutional layer 4: 64 filters, 3 × 3
        self.pool_2 = nn.MaxPool2d(2, 2) # Max-pooling layer 2: 2 × 2 windows

        self.fc1 = nn.Linear(1600, 512) # Fully connected layer 1: 512 units
        self.fc2 = nn.Linear(512, 10) # Output layer with num_classes
        
        self.dropout = nn.Dropout(0.5) # add Dropout layer


    def forward(self, x):

        x = F.relu(self.conv_1(x))      # conv_1 
        
        x = self.pool_1(F.relu(self.conv_2(x))) # conv_2 + pool_1
        
        x = self.dropout(x)
        
        x = F.relu(self.conv_3(x)) # conv_3
        
        x = self.pool_2(F.relu(self.conv_4(x))) # conv_4 + pool_2
        
        x = self.dropout(x)
        
        x = nn.Flatten()(x)

        x = F.relu(self.fc1(x)) # fully connected layer
        x = self.fc2(x) # fully connected layer + output layer
        return x


model = Conv_network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Training
train_loss_steps = []
train_acc_steps = []

train_loss_epochs = []
train_acc_epochs = []

valid_acc = []
valid_loss = []
for epoch in range(1, epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # put the model to train mode
        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = model(images)  # shape (B, 10).
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss = loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 500 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  #f'train_loss: {running_loss / run_step :.3f}, '
                  f'train_loss: {round(running_loss, 3)}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            
            train_loss_steps.append(round(running_loss, 3)) 
            train_acc_steps.append(round(running_correct / running_total, 3))
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
            
    train_loss_epochs.append(*train_loss_steps[-1:]) # add final value from each epoch 
    train_acc_epochs.append(*train_acc_steps[-1:])

    # validation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
#             images = images.view(-1, 32*32)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            validation_loss = loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        valid_loss.append(round(validation_loss, 3))
        valid_acc.append(round(correct / total, 3))
    print(f'Validation accuracy: {100 * correct / total: .1f} %')
    
print('Finished Training')




iters = np.arange(1,30,1) 
print(f'len(train_loss_epochs) - {len(train_loss_epochs)}')
print(f'len(train_acc_epochs) - {len(train_acc_epochs)}')
print(f'len(valid_acc) - {len(valid_acc)}')
print(f'len(valid_loss) - {len(valid_loss)}')



fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Dropout p = 0.5")
ax.plot(iters, train_loss_epochs, color='blue', label='Train')
ax.plot(iters, valid_loss, color='red', label='Valid')
ax.legend(loc='best')
ax.set_xlabel ("epochs", fontsize =16)
ax.set_ylabel ("CrossEntropyLoss", fontsize =16)


fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Dropout p = 0.5")
ax.plot(iters, train_acc_epochs, color='blue', label='Train')
ax.plot(iters, valid_acc, color='red', label='Valid')
ax.legend(loc='best')
ax.set_xlabel ("epochs", fontsize =16)
ax.set_ylabel ("Accuracy", fontsize =16)




## Dropout (p = 0.8) ------------------
class Conv_network(nn.Module):
    def __init__(self):
        super(Conv_network, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3) # Convolutional layer 1: 32 filters, 3 × 3
 
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3) # Convolutional layer 2: 32 filters, 3 × 3
      
        self.pool_1 = nn.MaxPool2d(2, 2) # Max-pooling layer 1: 2 × 2 windows
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3) # Convolutional layer 3: 64 filters, 3 × 3
     
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # Convolutional layer 4: 64 filters, 3 × 3
        self.pool_2 = nn.MaxPool2d(2, 2) # Max-pooling layer 2: 2 × 2 windows

        self.fc1 = nn.Linear(1600, 512) # Fully connected layer 1: 512 units
        self.fc2 = nn.Linear(512, 10) # Output layer with num_classes
        
        self.dropout = nn.Dropout(0.8) # add Dropout layer


    def forward(self, x):

        x = F.relu(self.conv_1(x))      # conv_1 
        
        x = self.pool_1(F.relu(self.conv_2(x))) # conv_2 + pool_1
        
        x = self.dropout(x)
        
        x = F.relu(self.conv_3(x)) # conv_3
        
        x = self.pool_2(F.relu(self.conv_4(x))) # conv_4 + pool_2
        
        x = self.dropout(x)
        x = nn.Flatten()(x)

        
        x = F.relu(self.fc1(x)) # fully connected layer

        x = self.fc2(x) # fully connected layer + output layer

        return x


model = Conv_network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Training
train_loss_steps = []
train_acc_steps = []

train_loss_epochs = []
train_acc_epochs = []

valid_acc = []
valid_loss = []
for epoch in range(1, epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # put the model to train mode
        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = model(images)  # shape (B, 10).
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss = loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 500 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  #f'train_loss: {running_loss / run_step :.3f}, '
                  f'train_loss: {round(running_loss, 3)}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            
            train_loss_steps.append(round(running_loss, 3)) 
            train_acc_steps.append(round(running_correct / running_total, 3))
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
            
    train_loss_epochs.append(*train_loss_steps[-1:]) # add final value from each epoch 
    train_acc_epochs.append(*train_acc_steps[-1:])

    # validation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
#             images = images.view(-1, 32*32)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            validation_loss = loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        valid_loss.append(round(validation_loss, 3))
        valid_acc.append(round(correct / total, 3))
        #valid_loss.append(1 - round(correct/total,3))
    print(f'Validation accuracy: {100 * correct / total: .1f} %')
    
print('Finished Training')




iters = np.arange(1,30,1) 
print(f'len(train_loss_epochs) - {len(train_loss_epochs)}')
print(f'len(train_acc_epochs) - {len(train_acc_epochs)}')
print(f'len(valid_acc) - {len(valid_acc)}')
print(f'len(valid_loss) - {len(valid_loss)}')



fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Dropout p = 0.5")
ax.plot(iters, train_loss_epochs, color='blue', label='Train')
ax.plot(iters, valid_loss, color='red', label='Valid')
ax.legend(loc='best')
ax.set_xlabel ("epochs", fontsize =16)
ax.set_ylabel ("CrossEntropyLoss", fontsize =16)




fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Dropout p = 0.5")
ax.plot(iters, train_acc_epochs, color='blue', label='Train')
ax.plot(iters, valid_acc, color='red', label='Valid')
ax.legend(loc='best')
ax.set_xlabel ("epochs", fontsize =16)
ax.set_ylabel ("Accuracy", fontsize =16)




## 7 EXR (trying to reach 80% accuracy)

## First try -----------------------------
batch_size = 22
num_classes = 10
epochs = 35
learning_rate = 0.001
momentum = 0.9


class Conv_network(nn.Module):
    def __init__(self):
        super(Conv_network, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3) # Convolutional layer 1: 32 filters, 3 × 3
 
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3) # Convolutional layer 2: 32 filters, 3 × 3
      
        self.pool_1 = nn.MaxPool2d(2, 2) # Max-pooling layer 1: 2 × 2 windows
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3) # Convolutional layer 3: 64 filters, 3 × 3
     
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # Convolutional layer 4: 64 filters, 3 × 3
        self.pool_2 = nn.MaxPool2d(2, 2) # Max-pooling layer 2: 2 × 2 windows

        self.fc1 = nn.Linear(1600, 512) # Fully connected layer 1: 512 units
        self.fc2 = nn.Linear(512, 10) # Output layer with num_classes
        
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):

        x = F.relu(self.conv_1(x))      # conv_1 
        
        x = self.pool_1(F.relu(self.conv_2(x))) # conv_2 + pool_1
        
        x = self.dropout(x)
        
        x = F.relu(self.conv_3(x)) # conv_3
        
        x = self.pool_2(F.relu(self.conv_4(x))) # conv_4 + pool_2
        
        x = self.dropout(x)
        

        x = nn.Flatten()(x)

        x = F.relu(self.fc1(x)) # fully connected layer

        x = self.fc2(x) # fully connected layer + output layer

        return x



model = Conv_network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)



# Training
train_loss_steps = []
train_acc_steps = []

train_loss_epochs = []
train_acc_epochs = []

valid_acc = []
valid_loss = []
for epoch in range(1, epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # put the model to train mode
        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = model(images)  # shape (B, 10).
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss = loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 500 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  #f'train_loss: {running_loss / run_step :.3f}, '
                  f'train_loss: {round(running_loss, 3)}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            
            train_loss_steps.append(round(running_loss, 3)) 
            #train_loss_steps.append(round(running_loss / run_step, 3))  # add values in each 500 steps for checking during training
            train_acc_steps.append(round(running_correct / running_total, 3))
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
            
    train_loss_epochs.append(*train_loss_steps[-1:]) # add final value from each epoch 
    train_acc_epochs.append(*train_acc_steps[-1:])

    # validation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
#             images = images.view(-1, 32*32)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            validation_loss = loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        valid_loss.append(round(validation_loss, 3))
        valid_acc.append(round(correct / total, 3))
        #valid_loss.append(1 - round(correct/total,3))
    print(f'Validation accuracy: {100 * correct / total: .1f} %')
    
print('Finished Training')



correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        validation_loss = loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total

print(f'Test accuracy: {100 * correct / total: .1f} %')



## Secind try ----------------

batch_size = 16
num_classes = 10
epochs = 40
learning_rate = 0.001
momentum = 0.9


model = Conv_network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Training
train_loss_steps = []
train_acc_steps = []

train_loss_epochs = []
train_acc_epochs = []

valid_acc = []
valid_loss = []
for epoch in range(1, epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # put the model to train mode

        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = model(images)  # shape (B, 10).
        loss = loss_fn(outputs, labels)
        #train_loss.append(loss)
        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss = loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 500 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  #f'train_loss: {running_loss / run_step :.3f}, '
                  f'train_loss: {round(running_loss, 3)}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            
            train_loss_steps.append(round(running_loss, 3)) 

            train_acc_steps.append(round(running_correct / running_total, 3))
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
            
    train_loss_epochs.append(*train_loss_steps[-1:]) # add final value from each epoch 
    train_acc_epochs.append(*train_acc_steps[-1:])

    # validation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
#             images = images.view(-1, 32*32)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            validation_loss = loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        valid_loss.append(round(validation_loss, 3))
        valid_acc.append(round(correct / total, 3))

    print(f'Validation accuracy: {100 * correct / total: .1f} %')
    
print('Finished Training')



correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        validation_loss = loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
    #valid_loss.append(round(validation_loss, 3))
    #valid_acc.append(round(correct / total, 3))
print(f'Test accuracy: {100 * correct / total: .1f} %')




## Trird try ---------------------------

batch_size = 10
num_classes = 10
epochs = 50
learning_rate = 0.001
momentum = 0.9



class Conv_network(nn.Module):
    def __init__(self):
        super(Conv_network, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3) # Convolutional layer 1: 32 filters, 3 × 3
 
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3) # Convolutional layer 2: 32 filters, 3 × 3
      
        self.pool_1 = nn.MaxPool2d(2, 2) # Max-pooling layer 1: 2 × 2 windows
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3) # Convolutional layer 3: 64 filters, 3 × 3
     
        self.conv_4 = nn.Conv2d(128, 256, kernel_size=3) # Convolutional layer 4: 64 filters, 3 × 3
        
        self.pool_2 = nn.MaxPool2d(2, 2) # Max-pooling layer 2: 2 × 2 windows

        self.fc1 = nn.Linear(256*5*5, 1024) # Fully connected layer 1: 512 units
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10) # Output layer with num_classes
        
        self.dropout = nn.Dropout(0.2)



    def forward(self, x):

        x = F.relu(self.conv_1(x))      # conv_1 
        
        x = self.pool_1(F.relu(self.conv_2(x))) # conv_2 + pool_1
        
        x = self.dropout(x)
        
        x = F.relu(self.conv_3(x)) # conv_3
        
        x = self.pool_2(F.relu(self.conv_4(x))) # conv_4 + pool_2

        
        x = self.dropout(x)
        
        x = nn.Flatten()(x)

        x = F.relu(self.fc1(x)) # fully connected layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # fully connected layer + output layer
        return x



model = Conv_network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




# Training
train_loss_steps = []
train_acc_steps = []

train_loss_epochs = []
train_acc_epochs = []

valid_acc = []
valid_loss = []
for epoch in range(1, epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # put the model to train mode
        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = model(images)  # shape (B, 10).
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss = loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 500 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  #f'train_loss: {running_loss / run_step :.3f}, '
                  f'train_loss: {round(running_loss, 3)}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            
            train_loss_steps.append(round(running_loss, 3)) 
            #train_loss_steps.append(round(running_loss / run_step, 3))  # add values in each 500 steps for checking during training
            train_acc_steps.append(round(running_correct / running_total, 3))
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
            
    train_loss_epochs.append(*train_loss_steps[-1:]) # add final value from each epoch 
    train_acc_epochs.append(*train_acc_steps[-1:])

    # validation
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
#             images = images.view(-1, 32*32)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            validation_loss = loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total

        valid_loss.append(round(validation_loss, 3))
        valid_acc.append(round(correct / total, 3))
    print(f'Validation accuracy: {100 * correct / total: .1f} %')
    
print('Finished Training')



correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        validation_loss = loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
print(f'Test accuracy: {100 * correct / total: .1f} %')    ## Test accuracy:  79.3 %




