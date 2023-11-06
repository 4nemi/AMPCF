import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from datafetcher import DataFetcher
from ampcf import AMPCF


class Config:
    num_users: int
    num_items: int
    num_personas: int = 4
    embedding_dim: int = 64
    attention_dim: int = 64

    alpha: float = 0.5
    lambda_p: float = 1.0
    lambda_n: float = 1.0

    lr: float = 0.001
    epochs: int = 2


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    datafetcher = DataFetcher(num_test_items=5, dataset_name="ml-1m")
    datacontainer = datafetcher.load()
    train_loader = DataLoader(
        datacontainer.train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )

    Config.num_users = len(np.unique(datacontainer.train.user_ids))
    Config.num_items = len(np.unique(datacontainer.train.item_ids))

    model = AMPCF(
        Config.num_users,
        Config.num_items,
        Config.num_personas,
        Config.embedding_dim,
        Config.attention_dim,
    )

    model.to(device)

    model.train()
    optimizer = Adam(model.parameters(), lr=Config.lr)

    best_score = 2**10

    for epoch in range(Config.epochs):
        for step, (user_ids, item_ids) in enumerate(train_loader):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)

            optimizer.zero_grad()

            loss = model.cal_loss(user_ids, item_ids, 4, Config.lambda_p, Config.lambda_n, Config.alpha)

            loss.backward()
            optimizer.step()

            if step % 100 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch: [{epoch+1}]/[{step}/{len(train_loader)}] Loss {loss.item()}")

        # TODO: Implement validation

        if best_score > loss.item():
            best_score = loss.item()
            torch.save(model.state_dict(), "../models/model.pt")

        print(f"Best score: {best_score}")


if __name__ == "__main__":
    train()
