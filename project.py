import torch
import torch.nn as nn
from training_loop import train
import torch.backends.cudnn as cudnn
from NN_DL import ConvNeuralNet, data_load

import warnings
warnings.filterwarnings("ignore")
    
# Enable cudnn benchmark mode
cudnn.benchmark = True

def main():
    # ----------------------- Data Loading & Model Setup ---------------------------------
    learning_rate = 0.0005
    num_epochs = 10
    batch_size = 8
    wd = 0.001

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_loader, val_loader, test_loader = data_load(params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvNeuralNet().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01, patience=3)

    # ----------------------- Training ---------------------------------

    history, best_weights = train(num_epochs, train_loader, val_loader, device, scheduler, criterion, optimizer, model)

    # ----------------------- Testing ---------------------------------
    torch.save(best_weights, './model.pth')
    model.load_state_dict(best_weights)

    with torch.no_grad():
        mse = 0.0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            mse += criterion(outputs, labels).item() * images.size(0)
        
        mse /= batch_size * len(test_loader)
        print('Mean Squared Error on the test set: {:.3f}'.format(mse))

if __name__ == '__main__':
    main()