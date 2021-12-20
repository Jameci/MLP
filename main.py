import train
import dataloader


if __name__ == '__main__':
    train_set, test_set, cross_set = dataloader.load_data()
    train.train(train_set, 10, 100)
