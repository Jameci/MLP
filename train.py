import net
import dataset
import tqdm


def train(data: dataset.Dataset, t: int, batch: int):
    res = net.Net()
    for i in range(t):
        loss = 0
        acc = 0
        for j in tqdm.tqdm(range(data.length // batch)):
            x = data.x[j * batch: j * batch + batch]
            y = data.y[j * batch: j * batch + batch]
            for k in range(batch):
                res.x = x[k]
                res.y = y[k]
                res.forward()
                res.backward()
                loss += res.loss()
                acc += res.acc()
            res.update(batch)
            res.set_zero()
        print(loss)
        print(acc)
    return res
