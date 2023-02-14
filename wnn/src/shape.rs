use std::fmt::Display;

use anyhow::bail;

use crate::onnx::tensor_shape_proto;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Dimension {
    Symbolic(String),
    Concrete(usize),
    Map,
    Rest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<Dimension>,
}

impl Shape {
    pub fn from(dims: &[i64]) -> Self {
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

    pub(crate) fn from_tensor_shape_with_maps(
        proto: &crate::onnx::TensorShapeProto,
        map: &std::collections::HashMap<&str, Dimension>,
    ) -> Self {
        let dims = proto
            .dim
            .iter()
            .map(|dim| match &dim.value {
                Some(tensor_shape_proto::dimension::Value::DimValue(i)) => {
                    Dimension::Concrete(*i as _)
                }
                Some(tensor_shape_proto::dimension::Value::DimParam(content)) => {
                    match map.get(content.as_str()) {
                        Some(dim) => dim.clone(),
                        None => Dimension::Symbolic(content.clone()),
                    }
                }
                None => unreachable!("dimension = None"),
            })
            .collect();
        Self { dims }
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

    pub(crate) fn reldim(&self, dim: isize) -> usize {
        if dim < 0 {
            (self.dims.len() as isize + dim) as usize
        } else {
            dim as usize
        }
    }

    pub fn is_concrete(&self) -> bool {
        self.dims
            .iter()
            .all(|d| matches!(d, Dimension::Concrete(_)))
    }

    pub(crate) fn size(&self, dim: isize) -> &Dimension {
        &self.dims[self.reldim(dim)]
    }

    pub(crate) fn as_ints(&self) -> anyhow::Result<Vec<i64>> {
        (0..self.ndims()).map(|i| self.as_int(i as _)).collect()
    }

    // TODO: Store constants as Vec<i64> not shapes.
    pub(crate) fn as_int(&self, dim: isize) -> anyhow::Result<i64> {
        match self.dims[self.reldim(dim)] {
            Dimension::Concrete(d) => Ok(d as i64),
            Dimension::Map => Ok(0),
            Dimension::Rest => Ok(-1),
            _ => Err(anyhow::anyhow!(
                "dimension {} is not concrete for shape {self}",
                self.reldim(dim)
            )),
        }
    }

    pub(crate) fn concrete_size(&self, dim: isize) -> anyhow::Result<usize> {
        match self.dims[self.reldim(dim)] {
            Dimension::Concrete(d) => Ok(d),
            _ => Err(anyhow::anyhow!(
                "dimension {} is not concrete for shape {self}",
                self.reldim(dim)
            )),
        }
    }

    pub(crate) fn set_dim(&mut self, dim: isize, val: Dimension) {
        let index = self.reldim(dim);
        self.dims[index] = val;
    }

    pub(crate) fn ndims(&self) -> usize {
        self.dims.len()
    }

    pub(crate) fn numel(&self) -> Option<usize> {
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

        if let (Some(other_prod), true) = (
            other.numel(),
            dims.iter()
                .all(|dim| !matches!(dim, Dimension::Symbolic(_) | Dimension::Map))
                && dims
                    .iter()
                    .filter(|dim| matches!(dim, Dimension::Rest))
                    .count()
                    == 1,
        ) {
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
                    break;
                }
            }
        }

        Self { dims }
    }

    pub(crate) fn squeeze(&mut self) {
        self.dims.retain(|d| !matches!(d, Dimension::Concrete(1)))
    }

    pub(crate) fn unsqueeze(&mut self, dim: usize) {
        self.dims.insert(dim, Dimension::Concrete(1))
    }

    pub(crate) fn transpose(&mut self, dim1: isize, dim2: isize) {
        let dim1 = self.reldim(dim1);
        let dim2 = self.reldim(dim2);
        self.dims.swap(dim1, dim2);
    }

    pub(crate) fn is_scalar(&self) -> bool {
        self.ndims() == 0
    }

    pub(crate) fn append_dim(&mut self, dim: Dimension) {
        self.dims.push(dim);
    }

    pub(crate) fn pad_left_to(&mut self, n: usize) {
        while self.ndims() < n {
            self.unsqueeze(0);
        }
    }

    pub(crate) fn permute(&mut self, perm: &[i64]) {
        self.dims = perm
            .iter()
            .map(|i| self.dims[*i as usize].clone())
            .collect();
    }

    pub(crate) fn broadcast(&mut self, a: &Shape) -> anyhow::Result<()> {
        for (bdim, adim) in self.dims.iter_mut().zip(&a.dims) {
            let new_val = match (&bdim, adim) {
                (Dimension::Concrete(1), _) => Some(adim.clone()),
                (_, Dimension::Concrete(1)) => None,
                _ if bdim != adim => {
                    anyhow::bail!("invalid broadcast between {self} and {a}");
                }
                _ => None,
            };
            if let Some(val) = new_val {
                *bdim = val;
            }
        }
        Ok(())
    }

    pub(crate) fn scale(&mut self, scales: &[f32]) -> anyhow::Result<()> {
        for (dim, s) in self.dims.iter_mut().zip(scales) {
            if let Dimension::Concrete(val) = dim {
                *dim = Dimension::Concrete((*s * *val as f32) as usize);
            } else if *s != 1. {
                bail!("unsupported scaling {} on dimension {:?}", s, &dim);
            }
        }
        Ok(())
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
            if self.ndims() == 1 || i != self.dims.len() - 1 {
                f.write_str(",")?;
            }
        }
        f.write_str(")")?;
        Ok(())
    }
}
