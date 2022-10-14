use anyhow::bail;

use crate::{onnx, shape::Shape};

pub(crate) enum DataType {
    I32,
    I64,
    F32,
    F64,
}

impl DataType {
    pub(crate) fn from_onnx(dtype: onnx::tensor_proto::DataType) -> anyhow::Result<Self> {
        Ok(match dtype {
            onnx::tensor_proto::DataType::INT32 => DataType::I32,
            onnx::tensor_proto::DataType::INT64 => DataType::I64,
            onnx::tensor_proto::DataType::FLOAT => DataType::F32,
            onnx::tensor_proto::DataType::DOUBLE => DataType::F64,
            _ => bail!("unsupported datatype {:?}", dtype),
        })
    }
}

pub(crate) struct Tensor {
    shape: Shape,
    dtype: DataType,
}

impl Tensor {
    pub(crate) fn new(shape: Shape, dtype: DataType) -> Self {
        Self { shape, dtype }
    }
}
