def fit_forward_split(cal, ts, p_pred, y, ts_train_end, ts_test_start):
    mask_calib = (ts >= ts_train_end) & (ts < ts_test_start)
    cal.fit(p_pred[mask_calib], y[mask_calib])
    return cal
