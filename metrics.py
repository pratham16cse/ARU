import numpy as np

def squaredError(y_act, y_pred):
    assert y_act.shape == y_pred.shape
    return np.square(y_act - y_pred)

def RMSE(y_act, y_pred):
    assert y_act.shape == y_pred.shape
    # sse = np.sum(squaredError(y_act, y_pred))
    # rmse = np.sqrt(sse * 1.0 / (y_act.shape[0] * y_act.shape[1]))
    rmse = np.sqrt(np.mean(squaredError(y_act, y_pred)))
    return rmse
   
def NRMSE(y_act, y_pred):
    assert y_act.shape == y_pred.shape
    rmse = RMSE(y_act, y_pred)
    # nrmse = rmse / ((1.0 / y_act.shape[0] * y_act.shape[1]) * np.sum(np.abs(y_act)))
    nrmse = rmse / np.mean(np.abs(y_act))
    return nrmse

def ND(y_act, y_pred): # Evaluation metric from DeepAR, equivalent to QL(rho=0.5)
    assert y_act.shape == y_pred.shape
    ae = np.sum(np.abs(y_act - y_pred))
    nd = ae / np.sum(np.abs(y_act))
    return nd

def QL(y_act, y_pred, rho=0.5, sigma=0.00001, num_samples=None): # Evaluation metric from DeepState
    assert y_act.shape == y_pred.shape
    if rho != 0.5:
        y_pred_quantile = 1.28 * sigma + y_pred
    else:
        y_pred_quantile = y_pred
    Z = np.abs(y_act - y_pred_quantile)
    rho_mat = rho * (y_act > y_pred_quantile) + (1.0 - rho) * (y_act <= y_pred_quantile)
    qloss = 2 * np.sum(Z * rho_mat) / np.sum(np.abs(y_act))
    return qloss

def QL_sample(y_act, y_pred, rho=0.5, sigma=0.0001, num_samples=1.0): # Evaluation metric from DeepState
    assert y_act.shape == y_pred.shape
    if rho != 0.5:
        y_pred_samples = list()
        for i in range(num_samples):
            sample = np.random.normal(loc=y_pred, scale=sigma)
            y_pred_samples.append(sample)
        y_pred_samples = np.stack(y_pred_samples, axis=2)
        y_pred_quantile = np.percentile(y_pred_samples, rho*100.0, axis=2)
    else:
        y_pred_quantile = y_pred
    Z = np.abs(y_act - y_pred_quantile)
    rho_mat = rho * (y_act > y_pred_quantile) + (1.0 - rho) * (y_act <= y_pred_quantile)
    qloss = 2 * np.sum(Z * rho_mat) / np.sum(np.abs(y_act))
    return qloss

def per_ts_ND(y_act, y_pred):
    assert y_act.shape == y_pred.shape
    ae = np.sum(np.abs(y_act - y_pred), axis=1)
    nd = ae / np.sum(np.abs(y_act), axis=1)
    assert y_act.shape[0] == nd.shape[0]
    return nd

def rho_risk(y_act, y_pred, rho=0.5): # Evaluation metric from DeepAR
    Z_act = np.sum(y_act, axis=1)
    Z_pred = np.sum(y_pred, axis=1)

    L_rho = 2 * (Z_pred - Z_act) (rho * (Z_pred > Z_act) - (1 - rho) * (Z_pred <= Z_act))
    L_rho = np.sum(L_rho)*1.0 / np.sum(Z_act)
    return L_rho
