use std::fmt::Display;

use crate::onnx::tensor_shape_proto;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Dimension {
    Symbolic(String),
    Concrete(usize),
    Map,
    Rest,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Shape {
    dims: Vec<Dimension>,
}

impl Shape {
    pub(crate) fn from(dims: &[i64]) -> Self {
        Self {
            dims: dims
                .iter()
                .map(|i| match i {
                    0 => Dimension::Map,
                    -1 => Dimension::Rest,
                    _ if *i < 0 => panic!("invalid dim {i}"),
                    _ => Dimension::Concrete(*i as _),
                })
                .collect(),
        }
    }

    pub(crate) fn from_tensor_shape(proto: &crate::onnx::TensorShapeProto) -> Self {
        let dims = proto
            .dim
            .iter()
            .map(|dim| match &dim.value {
                Some(tensor_shape_proto::dimension::Value::DimValue(i)) => {
                    Dimension::Concrete(*i as _)
                }
                Some(tensor_shape_proto::dimension::Value::DimParam(content)) => {
                    Dimension::Symbolic(content.clone())
                }
                None => unreachable!("dimension = None"),
            })
            .collect();
        Self { dims }
    }

    pub(crate) fn empty() -> Self {
        Self { dims: Vec::new() }
    }

    pub(crate) fn add_dim(&mut self, idx: isize, dim: Dimension) {
        self.dims.insert(self.reldim(idx), dim);
    }

    fn reldim(&self, dim: isize) -> usize {
        if dim < 0 {
            (self.dims.len() as isize + dim) as usize
        } else {
            dim as usize
        }
    }

    fn is_concrete(&self) -> bool {
        self.dims.iter().all(|d| match d {
            Dimension::Concrete(_) => true,
            _ => false,
        })
    }

    pub(crate) fn size(&self, dim: isize) -> &Dimension {
        &self.dims[self.reldim(dim)]
    }

    pub(crate) fn ndims(&self) -> usize {
        self.dims.len()
    }

    fn numel(&self) -> Option<usize> {
        let mut s = 1;
        for d in &self.dims {
            match d {
                Dimension::Concrete(i) => s *= i,
                _ => return None,
            }
        }
        Some(s)
    }

    pub(crate) fn map_and_rest(&self, other: &Shape) -> Self {
        let mut dims: Vec<Dimension> = self
            .dims
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if matches!(a, Dimension::Map) && i < other.dims.len() {
                    other.dims[i].clone()
                } else {
                    a.clone()
                }
            })
            .collect();

        match (
            other.numel(),
            dims.iter()
                .all(|dim| !matches!(dim, Dimension::Symbolic(_) | Dimension::Map))
                && dims
                    .iter()
                    .filter(|dim| matches!(dim, Dimension::Rest))
                    .count()
                    == 1,
        ) {
            (Some(other_prod), true) => {
                let self_prod: usize = dims
                    .iter()
                    .map(|dim| match dim {
                        Dimension::Concrete(i) => *i,
                        _ => 1,
                    })
                    .product();
                for dim in dims.iter_mut() {
                    if let Dimension::Rest = dim {
                        *dim = Dimension::Concrete(other_prod / self_prod);
                    }
                }
            }
            _ => {}
        }

        Self { dims }
    }

    pub(crate) fn squeeze(&mut self) {
        self.dims.retain(|d| match d {
            Dimension::Concrete(1) => false,
            _ => true,
        })
    }

    pub(crate) fn unsqueeze(&mut self, dim: usize) {
        self.dims.insert(dim, Dimension::Concrete(1))
    }

    pub(crate) fn transpose(&self, dim1: isize, dim2: isize) -> Self {
        let mut dims = self.dims.clone();
        dims.swap(self.reldim(dim1), self.reldim(dim2));
        Self { dims }
    }

    pub(crate) fn is_scalar(&self) -> bool {
        self.ndims() == 0
    }

    pub(crate) fn append_dim(&mut self, dim: Dimension) {
        self.dims.push(dim);
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("(")?;
        for (i, dim) in self.dims.iter().enumerate() {
            match dim {
                Dimension::Concrete(i) => f.write_fmt(format_args!("{}", *i))?,
                Dimension::Symbolic(name) => f.write_str(name)?,
                Dimension::Map => f.write_str("0")?,
                Dimension::Rest => f.write_str("-1")?,
            };
            if i != self.dims.len() - 1 {
                f.write_str(",")?;
            }
        }
        f.write_str(")")?;
        Ok(())
    }
}
