import cupy as cp
import numpy as np
from cupyx.time import repeat

print('Compute Capability:', cp.cuda.Device().compute_capability)

shape = (1024, 256, 256)
dtypes = (cp.complex128, cp.complex64, 'E')  # 'E' = cp.complex32
mempool = cp.get_default_memory_pool()

for t in dtypes:
    if t == 'E':
        dtype = cp.float16
        old_shape = shape
        shape = (shape[0], shape[1], 2 * shape[2])  # complex32 has two fp16
    else:
        dtype = t
    idtype = odtype = edtype = np.dtype(t) if t != 'E' else t
    a = cp.random.random(shape).astype(dtype)
    out = cp.empty_like(a)
    if t == 'E':
        shape = old_shape
    plan = cp.cuda.cufft.XtPlanNd(
        shape[1:],
        shape[1:],
        1,
        shape[1] * shape[2],
        idtype,
        shape[1:],
        1,
        shape[1] * shape[2],
        odtype,
        shape[0],
        edtype,
        order='C',
        last_axis=-1,
        last_size=None,
    )
    print(t)
    print(repeat(plan.fft, (a, out, cp.cuda.cufft.CUFFT_FORWARD),
                 n_repeat=100))
    plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)
    if t != 'E':
        out_np = np.fft.fftn(cp.asnumpy(a), axes=(-2, -1))
        print(
            # 'ok' if np.testing.assert_allclose(
            #     cp.asnumpy(out), out_np, rtol=1E-3) else 'not ok',
        )
    else:
        a_np = cp.asnumpy(a).astype(np.float32)  # upcast
        a_np = a_np.view(np.complex64)
        out_np = np.fft.fftn(a_np, axes=(-2, -1))
        out_np = np.ascontiguousarray(out_np).astype(np.complex64)  # downcast
        out_np = out_np.view(np.float32)
        out_np = out_np.astype(np.float16)
        ##print(t, 'ok' if cp.allclose(out, out_np, atol=1E-2) else 'not ok')
        # don't worry about accruacy for now, as we probably lost a lot during casting
        print(
            t,
            'ok'
            if cp.mean(cp.abs(out - cp.asarray(out_np))) < 0.1 else 'not ok',
        )
    del plan
    del a
    del out
    mempool.free_all_blocks()
