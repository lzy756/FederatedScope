import numpy as np


def nnls_solve(A: np.ndarray,
               b: np.ndarray,
               max_iter: int = 500,
               tol: float = 1e-9) -> np.ndarray:
    """
    Lawson-Hanson style Non-Negative Least Squares solver implemented with
    NumPy only (fallback when SciPy is unavailable).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape
    x = np.zeros(n, dtype=np.float64)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)

    iter_count = 0
    while np.any(w > tol) and iter_count < max_iter:
        t = np.argmax(w)
        passive[t] = True

        while True:
            iter_count += 1
            Ap = A[:, passive]
            if Ap.size == 0:
                break

            try:
                zp, *_ = np.linalg.lstsq(Ap, b, rcond=None)
            except np.linalg.LinAlgError:
                zp = np.linalg.pinv(Ap) @ b

            z = np.zeros(n, dtype=np.float64)
            z[passive] = zp

            if np.all(z[passive] >= -tol):
                x = z.clip(min=0)
                break

            negative_idx = (z < 0) & passive
            alpha = np.min(x[negative_idx] /
                           (x[negative_idx] - z[negative_idx] + tol))
            alpha = np.clip(alpha, 0.0, 1.0)
            x = x + alpha * (z - x)
            passive[(np.abs(x) < tol) & passive] = False

            if iter_count >= max_iter:
                break

        w = A.T @ (b - A @ x)

    return np.clip(x, 0.0, None)
