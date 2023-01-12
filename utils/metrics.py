import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    #自動忽略掉nan的值
    return np.nanmean((pred-true)**2)

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def NMAE(pred, true):
    #自動忽略掉nan的值
    #print(len(np.abs((pred - true))))
    return np.nanmean(np.abs((pred - true)))


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    nmae = NMAE(pred,true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,nmae,mspe