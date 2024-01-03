import numpy as np


def filtering(y, m, C, G, F, W, V):
    """
    (t-1)期において、1期先（t期）のフィルタリングを行う関数
    such as:
        x_t = G_t * x_(t-1) + w_t, w_t ~ N(0, W_t) : 状態方程式
        y_t = F_t * x_t + v_t, v_t ~ N(0, V_t) : 観測方程式
    
    Params:
        y: 観測値 [時点t]
        m, C: 状態の平均, 共分散行列 [t-1]
        G, F, W, V: 状態遷移行列, 観測行列, 状態誤差の共分散行列, 観測誤差の共分散行列 [t]
    Returns:
        tuple
            フィルタリング分布の平均と共分散行列 m, C [t]
            一期先予測分布の平均と共分散行列 a, R [t]
            一期先予測尤度の平均と共分散行列 f, Q [t]
    """
    # 一期先予測分布
    a = G @ m
    R = G @ C @ G.T + W
    # 一期先予測尤度
    f = F @ a
    Q = F @ R @ F.T + V
    # カルマンゲイン
    K = R @ F.T @ np.linalg.inv(Q)
    # 状態の更新
    m = a + K @ (y - f)
    C = R - K @ F @ R
    f_scalar, Q_scalar = f.item(), Q.item()
    return m, C, a, R, f_scalar, Q_scalar

def smoothing(s, S, m, C, a, R, G):
    """
    (t+1)期のsとSからt期のsとSを求める（状態の平滑化分布を求める）
    
    Params:
        s, S: 平滑化分布の平均, 共分散行列 [t+1]
        m, C: 状態の平均, 共分散行列 [t]
        a, R: 一期先予測分布の平均と共分散行列 [t+1]
        G: 状態遷移行列 [t+1]
    Returns:
        tuple
        平滑化分布の平均, 共分散行列 s, S [t]
    """
    # 平滑化利得
    A = C @ G.T @ np.linalg.inv(R)
    # 平滑化された状態
    s = m + A @ (s - a)
    S = C + A @ (S - R) @ A.T
    return s, S

def reverse_loglik(w_v, dims, y, G, F, m0, C0):
    """
    w_vを与えると対数尤度の-1倍を返す関数
    """
    # 分散は負にはならないのでexpを取る
    W = np.eye(dims) * np.exp(w_v[0])
    V = np.array([1]).reshape((1, 1)) * np.exp(w_v[1])
    T = len(y)
    m, C = np.zeros((T, dims)), np.zeros((T, dims, dims))
    a, R = np.zeros((T, dims)), np.zeros((T, dims, dims))
    f, Q = np.zeros((T)), np.zeros((T))

    for t in range(0, T):
        _F = F[t].reshape((1, dims))
        if t == 0:
            m[t], C[t], a[t], R[t], f[t], Q[t] = filtering(y[t], m0, C0, G, _F, W, V)
        else:
            m[t], C[t], a[t], R[t], f[t], Q[t] = filtering(y[t], m[t-1], C[t-1], G, _F, W, V)

    loglik = (-1) * np.sum(np.log(Q)) / 2 - (np.sum((y - f)**2 / Q)) / 2
    return (-1)*loglik
