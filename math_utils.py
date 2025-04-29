"""utils for mathematical calculation"""
import numpy as np


def argsort_and_unique(x, thres=0, keep_small=True):
    """sort the 1-D araay and filter the slim difference values
    Args:
        x: 1-D array
        thres: difference of values lower than the thres will be filtered
        keep_small: if any values will be filtered, true to keep the smallest one, false to keep the largest one

    Returns:
        order: sorted order of x
        y: x after sorted

    Usage:
        # just sort the values
        >>> x = [1, 2, 3, 4, 5]
        >>> order, _ = argsort_and_unique(x)
        >>> order
        [0 1 2 3 4]

        # difference of values lower than the thres will be seemed as duplicated values
        # filter them, and then, sorted them
        >>> order, y = argsort_and_unique(x, thres=1)
        >>> order
        [0 0 1 1 2]
        >>> y   # default keep the smallest one, values of 2, 4 will be filtered
        [1 3 5]

        # if any values will be filtered, keep the largest one
        >>> _, y = argsort_and_unique(x, thres=1, keep_small=False)
        >>> y   # 'cause keep the bigger item, 1, 3 will be filtered
        [2 4 5]

        >>> x = [7, 5, 2, 1, 4]
        >>> order, y = argsort_and_unique(x, thres=1, keep_small=False)
        >>> order
        [2 1 0 0 1]
        >>> y
        [2 5 7]

    """
    x = np.array(x)
    arg = np.argsort(np.argsort(x))
    x = np.sort(x)

    order = []
    y = []
    i = 0
    while x.size:
        diff = x[1:] - x[0]
        keep = diff <= thres
        keep = np.append(True, keep)

        tmp = x[keep]

        if keep_small:
            y.append(tmp[0])
        else:
            y.append(tmp[-1])

        order += [i] * len(tmp)
        x = x[~keep]
        i += 1

    order = np.array(order)[arg]
    y = np.array(y)
    return order, y


def arg_order_sort_2D(x, key=None, **kwargs):
    """sort the 2-D araay following the columns index, different to `np.argsort()`
    Args:
        x (np.ndarray): 2-D array, (m, n)
        key (tuple): indexs fall in [0, n)
        **kwargs: kwargs for `argsort_and_unique()`

    Returns:
        arg (np.ndarray): 1-D array (m, )

    Usage:
        # sort the values by all the order keys
        >>> x = np.array([[1, 2, 2, 5, 4], [10, 9, 8, 9, 7]]).T

        >>> arg_order_sort_2D(x)
        [0 2 1 4 3]

        # sort by x[:, 1] firstly; sort by x[:, 0] secondly
        >>> arg_order_sort_2D(x, key=(1, 0))
        [4 2 1 3 0]

        >>> np.argsort(x, axis=0).T
        [[0 1 2 4 3]
         [4 2 1 3 0]]
    """
    x = np.array(x)
    if key is None:
        key = list(range(x.shape[-1]))

    orders = []
    for k in key:
        order, _ = argsort_and_unique(x[:, k], **kwargs)
        orders.append(order)

    orders = np.array(orders).T
    idx = np.array(range(x.shape[0])).reshape((-1, 1))
    orders = np.concatenate([orders, idx], axis=1)
    orders = sorted(orders, key=lambda x: [x[i] for i in range(len(key))])
    orders = np.array(orders)
    arg = orders[:, -1]
    return arg


def order_sort_2D(x, key=None, **kwargs):
    """sort the 2-D araay following the columns index, different to `np.sort()`
    Args:
        x (np.ndarray): 2-D array, (m, n)
        key (tuple): indexs fall in [0, n)
        **kwargs: kwargs for `argsort_and_unique()`

    Returns:
        y (np.ndarray): 2-D array (m, n), x after sorted

    Usage:
        # sort the values by all the order keys
        >>> x = np.array([[1, 2, 2, 5, 4], [10, 9, 8, 9, 7]]).T

        >>> order_sort_2D(x).T
        [[ 1  2  2  4  5]
         [10  8  9  7  9]]

        # sort by x[:, 1] firstly, sort by x[:, 0] secondly
        >>> order_sort_2D(x, key=(1, 0)).T
        [[ 4  2  2  5  1]
         [ 7  8  9  9 10]]

        >>> np.sort(x, axis=0).T
        [[ 1  2  2  4  5]
         [ 7  8  9  9 10]]

    """
    arg = arg_order_sort_2D(x, key=key, **kwargs)
    return x[arg]


def transpose(x):
    """transpose the list, same behaviour to `np.transpose()`
    Args:
        x (List[list]): 2-D list

    Usage:

    """
    return list(zip(*x))


def make_divisible(v: float, divisor: int, min_value=None) -> int:
    """ensures that all layers have a channel number that is divisible by divisor

    Usage:
        >>> make_divisible(8, 8)
        8
        >>> make_divisible(9, 8)
        16
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
