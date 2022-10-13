use crate::shape::Shape;

pub(crate) enum DataType {
    F32,
    F64,
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
