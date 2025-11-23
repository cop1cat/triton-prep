import numpy as np
import onnxruntime as ort


class OnnxTester:
    def test(self, model_path):
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        feeds = {inp.name: self._make_tensor(inp) for inp in session.get_inputs()}
        session.run(None, feeds)

    def _make_tensor(self, inp):
        dtype = self._dtype_from_onnx(inp.type)
        shape = self._resolve_shape(inp.shape)
        name = inp.name.lower()

        if dtype in (np.int32, np.int64):
            if "mask" in name:
                return np.ones(shape, dtype=dtype)
            return np.random.randint(1, 50, size=shape, dtype=dtype)

        if dtype == np.bool_:
            return np.ones(shape, dtype=dtype)

        return np.random.randn(*shape).astype(dtype)

    def _resolve_shape(self, shape):
        dims = []
        for idx, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                dims.append(dim)
            elif idx == 0:
                dims.append(2)
            else:
                dims.append(16)
        return tuple(dims) if dims else (1,)

    def _dtype_from_onnx(self, onnx_type: str):
        mapping = {
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
        }
        return mapping.get(onnx_type, np.float32)
