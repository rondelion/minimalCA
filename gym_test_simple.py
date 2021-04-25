"""Component Integration Test."""

from __future__ import print_function

import json
import argparse
from itertools import cycle, islice

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms

from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder

class FlatFileDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super(FlatFileDataset).__init__()
        self.path = path

    def parse_file(self, path):
        with open(path, 'r') as f:
            for line in f:
                tokens = torch.tensor([eval(line.rstrip())], dtype=torch.float64).float()
                yield from tokens
                
    def get_stream(self, path):
        return cycle(self.parse_file(path))

    def __iter__(self):
        return self.parse_file(self.path)
        return self.get_stream(self.path)
        
def train(args, model, device, train_loader, optimizer, epoch, writer):
  """Trains the model for one epoch."""
  model.train()
  for batch_idx, data in enumerate(train_loader):
    max_decoded = 0.0
    min_decoded = 0.0
    data = data.to(device)
    optimizer.zero_grad()
    decoded, output = model(data)
    
    for x in decoded:
        for y in x:
            val = y.item()
            if val > max_decoded:
                max_decoded = val
            if val < min_decoded:
                min_decoded = val
    
    loss = F.mse_loss(output, data)
    loss.backward()
    optimizer.step()

    writer.add_image('train/inputs', torchvision.utils.make_grid(data), batch_idx)
    writer.add_image('train/outputs', torchvision.utils.make_grid(output), batch_idx)
    writer.add_scalar('train/loss', loss, batch_idx)

    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} \tLoss: {:.6f}\tMax decoded: {:.6f}\tMin decoded: {:.6f}'.format(
          epoch, loss.item(), max_decoded, min_decoded))

    if args.dry_run:
        break

def test(model, device, test_loader, writer):
  """Evaluates the trained model."""
  model.eval()
  test_loss = 0

  with torch.no_grad():
    cnt = 0
    for batch_idx, data in enumerate(test_loader):
      data = data.to(device)
      _, output = model(data)

      writer.add_image('test/inputs', torchvision.utils.make_grid(data), batch_idx)
      writer.add_image('test/outputs', torchvision.utils.make_grid(output), batch_idx)

      test_loss += F.mse_loss(output, data).item()  # sum up batch loss
      cnt += 1

    test_loss /= cnt
    writer.add_scalar('test/avg_loss', test_loss, 0)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument('--config', type=str, default='minimal_CA.json', metavar='N',
                      help='Model configuration (default: minimal_CA.json')
  parser.add_argument('--epochs', type=int, default=1, metavar='N',
                      help='Number of training epochs (default: 1)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--dry-run', action='store_true', default=False,
                      help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', help='File path for saving the current Model')
  parser.add_argument('--dataset', help='Input dataset path')

  args = parser.parse_args()

  torch.manual_seed(args.seed)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  with open(args.config) as config_file:
    ca_config = json.load(config_file)

  if 'visual' in ca_config:
    config = ca_config['visual']

  kwargs = {'batch_size': config['batch_size']}

  if use_cuda:
    kwargs.update({
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    })

  writer = SummaryWriter()

  train_dataset = FlatFileDataset(args.dataset)
  test_dataset =  FlatFileDataset(args.dataset)

# determining input shape from the input dataset
  with open(args.dataset) as f:
      input_size = len(eval(f.readline()))
  input_shape = [-1, input_size]

  train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
  test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

  if config['model'] == 'SimpleAE':
    model = SimpleAutoencoder(input_shape, config['model_config']).to(device)
  else:
    raise NotImplementedError('Model not supported: ' + str(config['model']))

  optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch, writer)
    test(model, device, test_loader, writer)

  if args.save_model is not None:
    torch.save(model.state_dict(), args.save_model)

if __name__ == '__main__':
    main()
