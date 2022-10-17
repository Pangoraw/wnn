use std::fmt::Display;

use anyhow::bail;

use crate::{onnx, shape::Shape};

#[derive(Clone, Debug)]
pub(crate) enum DataType {
    I32,
    I64,
    F32,
    F64,
}

impl DataType {
    pub(crate) fn from_int(dtype: i32) -> anyhow::Result<Self> {
        Ok(match dtype {
            1 => DataType::F32,
            6 => DataType::I32,
            7 => DataType::I64,
            11 => DataType::F64,
            _ => bail!("unsupported datatype {dtype}"),
        })
    }

    pub(crate) fn from_onnx(dtype: onnx::tensor_proto::DataType) -> anyhow::Result<Self> {
        Ok(match dtype {
            onnx::tensor_proto::DataType::INT32 => DataType::I32,
            onnx::tensor_proto::DataType::INT64 => DataType::I64,
            onnx::tensor_proto::DataType::FLOAT => DataType::F32,
            onnx::tensor_proto::DataType::DOUBLE => DataType::F64,
            _ => bail!("unsupported datatype {:?}", dtype),
        })
    }

    pub(crate) fn size_of(&self) -> usize {
        match self {
            DataType::I32 => std::mem::size_of::<i32>(),
            DataType::I64 => std::mem::size_of::<i64>(),
            DataType::F32 => std::mem::size_of::<f32>(),
            DataType::F64 => std::mem::size_of::<f64>(),
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::F32 => "f32",
            DataType::F64 => "f64",
        };
        f.write_str(str)
    }
}
