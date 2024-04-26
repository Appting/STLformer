# Liuzhaoxi 2023/11/7 16:16

import numpy as np


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean

#
# def MAPE(v, v_):
#     '''
#     Mean absolute percentage error.
#     :param v: np.ndarray or int, ground truth.
#     :param v_: np.ndarray or int, prediction.
#     :return: int, MAPE averages on all elements of input.
#     '''
#     mask=np.not_equal(v,0)
#     mask=mask.astype('float32')
#     mask=mask/np.mean(mask)
#     mape=np.abs(v_-v)/v
#     mape=np.mean(mask*mape)
#     return mape

def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    # print(f"v------->{v}")
    # print(f"v_-------->{v_}")
    # print(f"v------->{v.shape}")
    # print(f"v_-------->{v_.shape}")
    # v1= v[0,0:206,0]
    # v2 = v_[0,0:206, 0]
    # print(f"v1------->{v1}")
    # print(f"v2-------->{v2}")


    # 添加检查，确保分母不为零
    denominator = np.where(v != 0, v, 1)

    mask = np.not_equal(v, 0)
    mask = mask.astype('float32')
    mask = mask / np.mean(mask)

    # 修改为使用分母的绝对值，以避免可能的负数
    mape = np.abs(v_ - v) / np.abs(denominator)

    # 处理可能的无效值
    mape = np.where(np.isnan(mape), 0, mape)

    mape = np.mean(mask * mape)
    return mape


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)
    print('y:{},y_:{}'.format(y.shape,y_.shape))
    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y=np.expand_dims(y,axis=-1)
        #y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
