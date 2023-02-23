use std::fmt::Display;

use anyhow::bail;

use crate::shape::Shape;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataType {
    I32,
    I64,
    F16,
    F32,
    F64,
}

impl DataType {
    pub fn from_int(dtype: i32) -> anyhow::Result<Self> {
        Ok(match dtype {
            1 => DataType::F32,
            6 => DataType::I32,
            7 => DataType::I64,
            10 => DataType::F16,
            11 => DataType::F64,
            _ => bail!("unsupported datatype {dtype}"),
        })
    }

    pub(crate) fn size_of(&self) -> usize {
        match self {
            DataType::I32 => std::mem::size_of::<i32>(),
            DataType::I64 => std::mem::size_of::<i64>(),
            DataType::F16 => std::mem::size_of::<f32>() / 2,
            DataType::F32 => std::mem::size_of::<f32>(),
            DataType::F64 => std::mem::size_of::<f64>(),
        }
    }

    pub(crate) fn to_str(&self) -> &'static str {
        match self {
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::F16 => "f16",
            DataType::F32 => "f32",
            DataType::F64 => "f64",
        }
    }
}

#[derive(Clone)]
pub struct TensorDesc {
    pub shape: Shape,
    pub dtype: DataType,
}

impl TensorDesc {
    pub fn new(shape: Shape, dtype: DataType) -> Self {
        Self { shape, dtype }
    }

    pub(crate) fn static_f32(dims: &[i64]) -> Self {
        Self {
            shape: Shape::from(dims),
            dtype: DataType::F32,
        }
    }

    pub fn size_of(&self) -> usize {
        self.shape.numel().unwrap() * self.dtype.size_of()
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_str())
    }
}

pub enum CPUTensorData {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub struct CPUTensor {
    pub desc: TensorDesc,
    pub data: CPUTensorData,
}

impl CPUTensor {
    pub fn raw_data(&self) -> &[u8] {
        match &self.data {
            CPUTensorData::I32(v) => bytemuck::cast_slice(v),
            CPUTensorData::I64(v) => bytemuck::cast_slice(v),
            CPUTensorData::F32(v) => bytemuck::cast_slice(v),
            CPUTensorData::F64(v) => bytemuck::cast_slice(v),
        }
    }

    pub(crate) fn new(desc: TensorDesc, data: &[u8]) -> Self {
        let data = match desc.dtype {
            DataType::I32 => {
                let data = bytemuck::cast_slice(data);
                CPUTensorData::I32(data.to_vec())
            }
            DataType::I64 => {
                let data = bytemuck::cast_slice(data);
                CPUTensorData::I64(data.to_vec())
            }
            DataType::F32 => {
                let data = bytemuck::cast_slice(data);
                CPUTensorData::F32(data.to_vec())
            }
            DataType::F64 => {
                let data = bytemuck::cast_slice(data);
                CPUTensorData::F64(data.to_vec())
            }
            _ => unimplemented!("{}", desc.dtype),
        };
        Self { desc, data }
    }
}
