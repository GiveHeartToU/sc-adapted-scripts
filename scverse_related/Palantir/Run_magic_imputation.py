import os
import gc
import tempfile
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix, save_npz, load_npz
from joblib import Parallel, delayed
from typing import Union
import psutil
import shutil
import time
from contextlib import contextmanager

@contextmanager
def robust_tempdir(prefix="magic_memmap_", dir="./", max_retries=3):
    tmp_dir = tempfile.mkdtemp(prefix=prefix, dir=dir)
    try:
        yield tmp_dir
    finally:
        for attempt in range(max_retries):
            try:
                shutil.rmtree(tmp_dir)
                break
            except OSError as e:
                print(f"[Retry {attempt+1}] Failed to remove temp directory: {e}")
                time.sleep(3)
        else:
            print(f"[Warning] Temp directory '{tmp_dir}' not fully cleaned.")
            print(f"[Info] You may manually clean: {tmp_dir} if needed.")

def print_mem(message=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"[MEM] {message} | Memory: {mem_gb:.2f} GB")

def _dot_chunk_npz(t_matrix, chunk_path, start, end, out_path):
    X_chunk = load_npz(chunk_path)
    result = t_matrix @ X_chunk  # sparse × sparse 或 sparse × dense
    out = np.lib.format.open_memmap(out_path, mode="r+")
    out[:, start:end] = result.toarray().astype(np.float32)
    del result, X_chunk, out
    gc.collect()

def run_magic_imputation_sparse(
    data: Union[np.ndarray, pd.DataFrame, sc.AnnData],
    dm_res: Union[dict, None] = None,
    n_steps: int = 3,
    chunk_size: int = 100,
    sim_key: str = "DM_Similarity",
    expression_key: str = None,
    imputation_key: str = "MAGIC_imputed_data",
    n_jobs: int = -1,
    verbose: bool = True
) -> Union[np.ndarray, pd.DataFrame, None]:

    if isinstance(data, sc.AnnData):
        if expression_key:
            if expression_key not in data.layers.keys():
                raise ValueError(f"expression_key '{expression_key}' not found in .layers.")
            X = data.layers[expression_key]
        else:
            X = data.X
        if dm_res is None:
            T = data.obsp[sim_key]
    elif isinstance(data, pd.DataFrame):
        X = csr_matrix(data.values)
    else:
        X = csr_matrix(data)

    if dm_res is not None:
        T = dm_res["T"]
    elif not isinstance(data, sc.AnnData):
        raise ValueError("dm_res is required if data is not AnnData")

    if verbose:
        print_mem("Prepare diffusion matrix T**n")
    T_steps = T.copy()
    for _ in range(n_steps - 1):
        T_steps = T_steps @ T
        gc.collect()
        if verbose:
            print(f"Completed diffusion step {_ + 1}/{n_steps}")
    T_steps = T_steps.astype(np.float32)

    if not issparse(X):
        X = csr_matrix(X)
    X = X.astype(np.float32)

    shape = X.shape
    chunks = np.append(np.arange(0, shape[1], chunk_size), [shape[1]])

    if verbose:
        print_mem("Writing sparse chunks")
    with robust_tempdir(prefix="magic_memmap_", dir="./") as tmp_dir:
        chunk_paths = []
        for i in range(len(chunks) - 1):
            start, end = chunks[i], chunks[i+1]
            subX = X[:, start:end]
            path = os.path.join(tmp_dir, f"chunk_{i}.npz")
            save_npz(path, subX)
            chunk_paths.append((path, start, end))

        out_path = os.path.join(tmp_dir, "imputed_memmap.npy")
        _ = np.lib.format.open_memmap(out_path, dtype=np.float32, mode="w+", shape=shape)

        if verbose:
            print_mem("Start parallel dot product on disk")
        Parallel(n_jobs=n_jobs)(
            delayed(_dot_chunk_npz)(T_steps, path, start, end, out_path)
            for path, start, end in chunk_paths
        )

        if verbose:
            print_mem("Finished dot products. Loading result")

        imputed_array = np.load(out_path)
        imputed_array[imputed_array < 1e-2] = 0  # optional threshold
        imputed_data = csr_matrix(imputed_array)
        del imputed_array
        gc.collect()

    if isinstance(data, sc.AnnData):
        data.layers[imputation_key] = imputed_data
        return None
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(imputed_data.toarray(), index=data.index, columns=data.columns)
    return imputed_data

# Monkey patch
import palantir
palantir.utils.run_magic_imputation = run_magic_imputation_sparse
print("MAGIC sparse version has been monkey patched into palantir.")

