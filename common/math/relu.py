def relu(x):
    if x > 0:
        return x
    else:
        return 0


def np_relu(x):
    return x * (x > 0)
