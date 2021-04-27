import random
from tqdm import tqdm
import torch
from cfg import parameters
from net import UnifiedNetwork
from dataset import UnifiedPoseDataset

from tensorboardX import SummaryWriter


if __name__ == '__main__':
    continue_train = False
    load_epoch = 0

    training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train')
    testing_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test')

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = parameters.batch_size, shuffle=True, num_workers=4)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size = parameters.batch_size, shuffle=False, num_workers=4)

    model = UnifiedNetwork()
    assert (torch.cuda.is_available())
    model.to(0)     #model.cuda()
    if continue_train:
        device = torch.device('cuda:0')
        state_dict = torch.load('../models/unified_net_originial.pth', map_location=str(device))
        model.load_state_dict(state_dict)
        print("load success")

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)

    best_loss = float('inf')

    writer = SummaryWriter()

    epoch_range = parameters.epochs
    if continue_train:
        epoch_range -= load_epoch

    for epoch in range(epoch_range):
        # train
        model.train()
        training_loss = 0.
        for batch, data in enumerate(tqdm(training_dataloader)):
            optimizer.zero_grad()
            image = data[0]
            true = [x.cuda() for x in data[1:]]

            if torch.isnan(image).any():
                raise ValueError('Image error')

            # At initial or if training dataset's sequence # changed, apply zero extra_keypoint
            # if not, extrapolate keypoint from previous pred

            pred = model(image.cuda())
            loss = model.total_loss(pred, true)
            training_loss += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()

        training_loss = training_loss / batch
        writer.add_scalars('data/loss', {'train_loss': training_loss}, epoch)

        # validation and save model
        if epoch % 10 == 0:
            validation_loss = 0.
            with torch.no_grad():
                for batch, data in enumerate(tqdm(testing_dataloader)):

                    image = data[0]
                    true = [x.cuda() for x in data[1:]]

                    if torch.isnan(image).any():
                        raise ValueError('WTF?!')

                    pred = model(image.cuda())
                    loss = model.total_loss(pred, true)
                    validation_loss += loss.data.cpu().numpy()

            validation_loss = validation_loss / batch
            writer.add_scalars('data/loss', {'val_loss': validation_loss}, epoch)
            print("Epoch : {} finished. Validation Loss: {}".format(epoch, validation_loss))
            torch.save(model.state_dict(), '../models/unified_net_originial.pth')

        print("Epoch : {} finished. Training Loss: {}.".format(epoch, training_loss))