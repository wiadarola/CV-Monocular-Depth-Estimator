import torch
import copy
from torch.cuda.amp import autocast
 
def train(num_epochs, train_loader, val_loader, device, scheduler, criterion, optimizer, model):
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    patience = 3  # Number of epochs to wait for improvement
    threshold = 1000
    counter = 0  # Counter to track the number of epochs without improvement
    best_weights = None
    history = []

    for epoch in range(1,num_epochs+1):
        running_loss = 0.0

        # Training
        for i, (local_batch, local_labels) in enumerate(train_loader, 1):
            # Move tensors to the configured device
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            optimizer.zero_grad()
            # Forward pass
            with autocast():
                outputs = model(local_batch)
                loss = criterion(outputs, local_labels)

            # Backward and optimize
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Free up GPU Memory cache
            torch.cuda.empty_cache()

        train_loss = running_loss / len(train_loader)
        print('Epoch [{}/{}], Training Loss: {:.4f}'.format(epoch, num_epochs, train_loss))

        # Validation
        with torch.no_grad():
            loss_sum = 0
            
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss_sum += criterion(outputs, labels).item()
            
            val_loss = loss_sum / len(val_loader)
            print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch, num_epochs, val_loss))
            history.append(("Epoch " + str(epoch), val_loss))
            
            # Early stopping
            if val_loss < best_val_loss - threshold:
                print("New best validation loss at epoch", epoch)
                best_val_loss = val_loss
                counter = 0  # Reset the counter when there is improvement
                best_weights = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping at epoch", epoch)
                    break
            scheduler.step(val_loss)
    return history, best_weights