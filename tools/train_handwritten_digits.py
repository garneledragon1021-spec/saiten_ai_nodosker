"""Train the small on-device digit fallback on MNIST; no user photos are training data.

Dataset loader/mirror: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
Original dataset: http://yann.lecun.com/exdb/mnist/
"""
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(20261019)
torch.set_num_threads(4)

class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(48 * 7 * 7, 96), nn.ReLU(), nn.Dropout(.2), nn.Linear(96, 10),
        )
    def forward(self, image):
        return self.layers(image)

def main():
    started = time.time()
    root = Path('.model-cache/mnist')
    training = datasets.MNIST(root, train=True, download=True, transform=transforms.Compose([
        transforms.RandomAffine(12, translate=(.08, .08), scale=(.85, 1.15)), transforms.ToTensor(),
    ]))
    testing = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(training, batch_size=192, shuffle=True, num_workers=0)
    test_loader = DataLoader(testing, batch_size=256, num_workers=0)
    model = DigitNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(4):
        model.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(image), label)
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch + 1}/4 complete, elapsed {time.time() - started:.1f}s', flush=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for image, label in test_loader:
            correct += int((model(image).argmax(1) == label).sum())
            total += len(label)
    target = Path('app/src/main/assets/handwritten_digits_v1.onnx')
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), target, input_names=['image'], output_names=['logits'],
                      opset_version=17, dynamo=False)
    metadata = dict(dataset='MNIST', training_images=len(training), held_out_images=total,
                    held_out_accuracy=correct / total, epochs=4, seed=20261019,
                    normalization='float32 [0,1], white foreground on black, NCHW 1x1x28x28',
                    limitations='MNIST test accuracy is not real-world score-sheet accuracy. User screenshots are evaluation only.')
    Path('tools/handwritten_model_metrics.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata), flush=True)

if __name__ == '__main__':
    main()
