import numpy as np
import torch

from collections import OrderedDict
from image_utils import process_image
from torch import nn, optim
from torchvision import models

def make_network(hidden_size, output_size, dropout, output_dim, model_architecture='vgg16'):
    print("making {} network...\nhiden_size: {}\noutput_size: {}\ndropout: {}\n".format(
        model_architecture, hidden_size, output_size, dropout))
    if model_architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif model_architecture == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
     # prevent modification of feature weights
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_size)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_size, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    if model_architecture == 'vgg16':
        model.classifier = classifier
    elif model_architecture == 'resnet18':
        model.fc = classifier
    return model

def train_network(model, dataloader=None, learnrate=0.001, criterion=None, optimizer=None,
    device=None, epochs=3, print_every=40):
    if criterion == None:
        criterion = nn.NLLLoss()
    if optimizer == None:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=learnrate)

    print("training network...\ndevice: {}\nlearnrate: {}\n".format(device, learnrate))

    steps = 0
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for _, (inputs, labels) in enumerate(dataloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
    return criterion, optimizer

def test_model(model, dataloader, device=None):
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {}%'.format(total, 100 * 
        (correct / total)))

def save_checkpoint(filename, model, arch, class_to_idx, optimizer, criterion, epochs=3,
    learnrate=0.001):
    if arch == 'vgg16':
        classifier = model.classifier
    elif arch == 'resnet18':
        classifier = model.fc
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'learnrate': learnrate,
        'epochs': epochs,
        'arch': {
          'relu': None,
          'dropout': classifier.dropout.p,
          'fc2': {'in': classifier.fc2.in_features, 'out': classifier.fc2.out_features},
          'output': {'dim': classifier.output.dim}
        },
        'class_to_idx': class_to_idx,
        'model_arch': arch
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    hidden_size = arch['fc2']['in']
    output_size = arch['fc2']['out']
    dropout = arch['dropout']
    output_dim = arch['output']['dim']
    model = make_network(hidden_size, output_size, dropout, output_dim, checkpoint['model_arch'])
    model.load_state_dict(checkpoint['model'])
    model.meta = {'class_to_idx': checkpoint['class_to_idx']}
    return model

def predict(image_path, model, means, stds, topk=5, device=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        print("getting top {} predictions using {}".format(topk, device))
        use_cpu = device == "cpu"
        model.to(device)
        image = torch.from_numpy(process_image(image_path, means, stds)).unsqueeze(0).to(device,
            torch.float)
        outputs = model(image)
        top = outputs.topk(topk)
        probs = top[0] if use_cpu else top[0].cpu()
        labels = top[1] if use_cpu else top[1].cpu()
        class_to_idx = model.meta.get("class_to_idx")
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        classes = [idx_to_class.get(label) for label in labels.numpy()[0]]
        return probs.numpy()[0], classes

def get_prediction_with_labels(model, image_path, image_to_name, means, stds, topk=5,
    device=None):
    probs, classes = predict(image_path, model, means, stds, topk, device)
    class_names = [image_to_name.get(label) for label in classes]
    # normalize probabilities
    exps = [np.exp(i) for i in probs]
    exps_sum = sum(exps)
    normalized_probs = np.array([i/exps_sum for i in exps])
    return class_names, probs, normalized_probs