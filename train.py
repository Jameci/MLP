import net
import dataset


def train(data: dataset.Dataset, t: int, batch: int):
    res = net.Net()
    for i in range(t):
        for j in range(data.length // batch):
            x = data.x[j * batch: j * batch + batch]
            y = data.y[j * batch: j * batch + batch]
            for k in range(batch):
                res.x = x[k]
                res.y = y[k]
                res.forward()
                res.backward()
            res.update(batch)
            res.set_zero()
    return res
