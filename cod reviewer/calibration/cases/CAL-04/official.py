"""Official training epoch (synthetic calibration case)."""


def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    for batch in loader:
        loss = loss_fn(model(batch['image']), batch['mask'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
