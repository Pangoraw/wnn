use std::collections::HashMap;

use anyhow::{anyhow, bail, Context};

use crate::shape::Shape;
use crate::utils::*;
use crate::{onnx, tensor::DataType};

use super::{LogicalOpType, PoolType};

pub(super) struct ShapeInferer<'a> {
    shapes: HashMap<&'a str, Shape>,
    constants: HashMap<&'a str, Vec<i64>>,
    known_shapes: HashMap<&'a str, Shape>,
    aliases: HashMap<&'a str, &'a str>,
    graph: &'a onnx::GraphProto,
}

impl<'a> ShapeInferer<'a> {
    pub(super) fn new(graph: &'a onnx::GraphProto) -> Self {
        Self {
            shapes: HashMap::new(),
            constants: HashMap::new(),
            known_shapes: HashMap::new(),
            aliases: HashMap::new(),
            graph,
        }
    }

    pub(super) fn infer_node(
        &mut self,
        node: &'a onnx::NodeProto,
    ) -> anyhow::Result<(super::LogicalOpType, Vec<Shape>)> {
        let out = match node.op_type() {
            op @ ("Gemm" | "MatMul") => {
                let trans_a = if op == "Gemm" {
                    node.attribute
                        .iter()
                        .find_map(|attr| {
                            if attr.name() == "transA" {
                                attr.i
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0)
                        == 1
                } else {
                    false
                };
                let trans_b = if op == "Gemm" {
                    node.attribute
                        .iter()
                        .find_map(|attr| {
                            if attr.name() == "transB" {
                                attr.i
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0)
                        == 1
                } else {
                    false
                };

                let input_shapes = node
                    .input
                    .iter()
                    .map(|input| {
                        self.shapes
                            .get(input.as_str())
                            .ok_or_else(|| anyhow!("failed to get input shape for {}", input))
                    })
                    .collect::<anyhow::Result<Vec<&Shape>>>()?;
                if (op == "Matmul" && input_shapes.len() != 2)
                    || (op == "Gemm" && (input_shapes.len() != 3 && input_shapes.len() != 2))
                {
                    bail!("invalid {op} with {}", input_shapes.len());
                }

                let mut a = input_shapes[0].clone();
                let mut b = input_shapes[1].clone();
                let c = input_shapes.get(2);

                let alpha = get_attr_float(node, "alpha").unwrap_or(1.);
                let beta = get_attr_float(node, "beta").unwrap_or(1.);

                if trans_a {
                    a.transpose(-1, -2)
                };
                if trans_b {
                    b.transpose(-1, -2)
                };

                if let Some(c) = c {
                    let mut c = (*c).clone();
                    c.squeeze(); // poor broadcast...
                    if !(c.ndims() == 1 && c.size(0) == b.size(-1)) {
                        bail!("invalid bias shape for Gemm c{}", c);
                    }
                }
                let mut out_dim = Shape::empty();
                if a.size(-1) == b.size(-2) {
                    for dim in 0isize..a.ndims() as isize - 1 {
                        out_dim.add_dim(dim, a.size(dim).clone());
                    }
                    out_dim.append_dim(b.size(-1).clone());
                } else {
                    bail!("invalid {} a{} @ b{}", op, a, b);
                }
                (
                    LogicalOpType::Gemm {
                        trans_a,
                        trans_b,
                        alpha,
                        beta,
                        activation: None,
                    },
                    vec![out_dim],
                )
            }
            op @ ("Mul" | "Pow" | "Add" | "Div" | "Sub") => {
                let input_shapes = node
                    .input
                    .iter()
                    .map(|input| {
                        self.shapes
                            .get(input.as_str())
                            .ok_or_else(|| anyhow!("failed to get shape for input {}", input))
                    })
                    .collect::<anyhow::Result<Vec<&Shape>>>()?;

                let mut a = input_shapes[0].clone();
                let mut b = input_shapes[1].clone();

                if let (Some(a), Some(b)) = (
                    self.find_constant(node.input[0].as_str()),
                    self.find_constant(node.input[1].as_str()),
                ) {
                    let c = a
                        .iter()
                        .zip(b)
                        .map(|(a, b)| match op {
                            "Mul" => a * b,
                            "Add" => a + b,
                            "Sub" => a - b,
                            "Div" => a / b,
                            "Pow" => a ^ b,
                            _ => 0,
                        })
                        .collect();
                    self.constants.insert(node.output[0].as_str(), c);
                }

                let out_shape = if a.is_scalar() {
                    b
                } else if b.is_scalar() {
                    a
                } else {
                    if a.ndims() > b.ndims() {
                        b.pad_left_to(a.ndims());
                    } else if b.ndims() > a.ndims() {
                        a.pad_left_to(b.ndims());
                    }

                    b.broadcast(&a)?;
                    b
                };

                (
                    match op {
                        "Mul" => LogicalOpType::Mul,
                        "Add" => LogicalOpType::Add,
                        "Sub" => LogicalOpType::Sub,
                        "Div" => LogicalOpType::Div,
                        "Pow" => LogicalOpType::Pow,
                        _ => unreachable!(),
                    },
                    vec![out_shape],
                )
            }
            "Expand" => {
                let mut out = self.shapes[node.input[0].as_str()].clone();
                let shape = self
                    .find_stored_shape(&node.input[1])
                    .ok_or_else(|| anyhow!("failed to find shape {}", node.input[1]))?;

                out.broadcast(&shape)?;
                (LogicalOpType::Expand, vec![out])
            }
            "Transpose" => {
                let perm = get_attr_ints(node, "perm")
                    .ok_or_else(|| anyhow!("invalid transpose: perm not found"))?;
                assert!(node.input.len() == 1);
                let mut shape = self.shapes[node.input[0].as_str()].clone();
                shape.permute(perm);

                (
                    LogicalOpType::Transpose {
                        perm: perm.to_vec(),
                    },
                    vec![shape],
                )
            }
            "Flatten" => {
                let input_shape = &self.shapes[node.input[0].as_str()];
                let mut output_shape = Shape::empty();
                output_shape.append_dim(input_shape.size(0).clone());
                output_shape.append_dim(crate::shape::Dimension::Rest.into());
                (
                    LogicalOpType::ReshapeOnly,
                    vec![output_shape.map_and_rest(input_shape)],
                )
            }
            "Concat" => {
                let input_shapes = node
                    .input
                    .iter()
                    .map(|input| {
                        self.shapes
                            .get(input.as_str())
                            .ok_or_else(|| anyhow!("failed to get shape for input {}", input))
                    })
                    .collect::<anyhow::Result<Vec<&Shape>>>()?;

                if input_shapes.iter().all(|shape| shape.ndims() == 1) {
                    if let Ok(constants) = node
                        .input
                        .iter()
                        .map(|input| {
                            self.find_constant(input)
                                .map(|s| s.to_owned())
                                .ok_or_else(|| anyhow!("could not find constant for {input}"))
                        })
                        .collect::<anyhow::Result<Vec<Vec<i64>>>>()
                    {
                        let new_constant =
                            constants.iter().flatten().copied().collect::<Vec<i64>>(); // Phew...
                        self.constants.insert(node.output[0].as_str(), new_constant);
                    }
                }

                let axis = get_attr_int(node, "axis").ok_or_else(|| anyhow!("no axis provided"))?;
                let mut out_shape = input_shapes[0].clone();
                out_shape.set_dim(
                    axis as isize,
                    crate::shape::Dimension::Concrete(
                        input_shapes
                            .iter()
                            .map(|shape| shape.concrete_size(axis as isize))
                            .collect::<anyhow::Result<Vec<usize>>>()?
                            .iter()
                            .sum(),
                    ),
                );

                (LogicalOpType::Concat { axis }, vec![out_shape])
            }
            "Squeeze" => {
                let mut input_shape = self.shapes[node.input[0].as_str()].clone();
                match node.input[1].as_str() {
                    "" => {}
                    other => bail!("unsupported dynamic Squeeze with input {other}"),
                }
                input_shape.squeeze();
                (LogicalOpType::ReshapeOnly, vec![input_shape])
            }
            "Unsqueeze" => {
                let mut out_shape = self.shapes[node.input[0].as_str()].clone();
                let axes = self
                    .find_stored_shape(&node.input[1])
                    .ok_or_else(|| anyhow!("failed to find shape for tensor {}", node.input[1]))?;

                if let Some(constant) = self.find_constant(node.input[0].as_str()) {
                    self.constants.insert(&node.output[0], constant.to_vec());
                }

                for dim in 0isize..axes.ndims() as isize {
                    out_shape.unsqueeze(axes.as_int(dim)? as usize);
                }

                (LogicalOpType::ReshapeOnly, vec![out_shape])
            }
            "Gather" => {
                let axis = get_attr_int(node, "axis").unwrap_or(0);
                let data = &self.shapes[node.input[0].as_str()];
                let indices = &self.shapes[node.input[1].as_str()];

                if let (Some(data), Some(index)) = (
                    self.find_constant(node.input[0].as_str()),
                    self.find_constant(node.input[1].as_str()),
                ) {
                    let out_const = index
                        .iter()
                        .map(|idx| {
                            if *idx < 0 {
                                data[(data.len() as i64 + *idx) as usize]
                            } else {
                                data[*idx as usize]
                            }
                        })
                        .collect();
                    self.constants.insert(node.output[0].as_str(), out_const);
                }

                let out_shape = if matches!(indices.numel(), Some(1)) {
                    let mut out_shape = Shape::empty();
                    for dim in 0..data.ndims() as i64 {
                        if dim == axis {
                            for index_dim in 0..indices.ndims() {
                                out_shape.append_dim(indices.size(index_dim as isize).clone())
                            }
                        } else {
                            out_shape.append_dim(data.size(dim as isize).clone())
                        }
                    }
                    out_shape
                } else {
                    unimplemented!();
                };
                (LogicalOpType::Gather { axis }, vec![out_shape])
            }
            op @ ("Conv" | "MaxPool") => {
                let input_shapes = node
                    .input
                    .iter()
                    .map(|input| {
                        self.shapes
                            .get(input.as_str())
                            .ok_or_else(|| anyhow!("failed to get shape for input {}", input))
                    })
                    .collect::<anyhow::Result<Vec<&Shape>>>()?;
                let strides = get_attr_ints(node, "strides").unwrap_or(&[1, 1]);
                let x = input_shapes[0];

                let kernel_shape = match (op, get_attr_ints(node, "kernel_shape")) {
                    (_, Some(shape)) => shape.to_owned(),
                    ("Conv", None) => {
                        let w = input_shapes[1];

                        if x.ndims() != 4 || w.ndims() != 4 {
                            bail!("invalid convolution {} o {}", x, w);
                        }

                        if x.concrete_size(1)? != w.concrete_size(1)? {
                            bail!("invalid convolution with input of size {x} and weights of size {w}");
                        }

                        vec![w.concrete_size(2)? as i64, w.concrete_size(3)? as i64]
                    }
                    _ => bail!("kernel_shape not found for {op}"),
                };

                let mut pads = [0; 4];
                let pads = match get_attr_ints(node, "pads") {
                    Some(pads) => pads,
                    None => match get_attr_string(node, "auto_pad") {
                        Some("SAME_UPPER") => {
                            pads[0] = kernel_shape[0] / 2;
                            pads[1] = kernel_shape[1] / 2;
                            pads[2] = kernel_shape[0] / 2;
                            pads[3] = kernel_shape[1] / 2;
                            &pads
                        }
                        Some("NOTSET") | None => bail!("no explicit padding specified for {op}"),
                        Some(other) => bail!("unsupported padding type {other} for {op}"),
                    },
                };

                let dilations = get_attr_ints(node, "dilations").unwrap_or(&[1, 1]);
                let (out_channels, pool_type) = match op {
                    "Conv" => {
                        let w = input_shapes[1];
                        (w.size(0).clone(), PoolType::Conv)
                    }
                    "MaxPool" => (x.size(1).clone(), PoolType::Max),
                    _ => unreachable!(),
                };

                if pads.len() != 4 {
                    bail!("invalid pads {:?}", pads);
                }

                let h_out = (x.concrete_size(2)? as i64 + 2 * pads[0]
                    - dilations[0] * (kernel_shape[0] - 1)
                    - 1)
                    / strides[0]
                    + 1;
                let w_out = (x.concrete_size(3)? as i64 + 2 * pads[1]
                    - dilations[1] * (kernel_shape[1] - 1)
                    - 1)
                    / strides[1]
                    + 1;

                let mut out_shape = Shape::empty();
                out_shape.append_dim(x.size(0).clone()); // B
                out_shape.append_dim(out_channels); // C
                out_shape.append_dim(crate::shape::Dimension::Concrete(h_out as usize).into()); // H
                out_shape.append_dim(crate::shape::Dimension::Concrete(w_out as usize).into()); // W

                (
                    LogicalOpType::Pool {
                        ptype: pool_type,
                        group: get_attr_int(node, "group").unwrap_or(1),
                        dilations: [dilations[0], dilations[1]],
                        k_strides: [strides[0], strides[1]],
                        pads: [pads[0], pads[1], pads[2], pads[3]],
                        kernel_shape: [kernel_shape[0], kernel_shape[1]],
                        activation: None,
                    },
                    vec![out_shape],
                )
            }
            "Resize" => {
                let scales = {
                    let node_name: &str = self
                        .aliases
                        .get(node.input[2].as_str())
                        .unwrap_or(&node.input[2].as_str());
                    self.graph
                        .initializer
                        .iter()
                        .find(|init| init.data_type() == 1 && init.name() == node_name)
                        .map(float_slice_from_tensor)
                        .ok_or_else(|| anyhow!("failed to get scales {}", node.input[2]))?
                };
                let mut input_shape = self.shapes[node.input[0].as_str()].clone();
                input_shape.scale(scales)?;

                (
                    LogicalOpType::Resize {
                        transformation_mode: match get_attr_string(
                            node,
                            "coordinate_transformation_mode",
                        ) {
                            Some("asymmetric") => {
                                super::ResizeCoordinateTransformationMode::Asymmetric
                            }
                            Some(other) => {
                                bail!("unsupported coordinate_transformation_mode '{}'", other)
                            }
                            None => {
                                bail!("unsupported coordinate_transformation_mode 'half_pixel'")
                            }
                        },
                    },
                    vec![input_shape],
                )
            }
            "Reshape" => {
                let shape = self
                    .find_stored_shape(&node.input[1])
                    .ok_or_else(|| anyhow!("failed to find shape for tensor {}", node.input[1]))?;
                let input_shape = &self.shapes[node.input[0].as_str()];
                let out_shape = shape.map_and_rest(input_shape);

                if out_shape.numel() != input_shape.numel() {
                    bail!("invalid Reshape {} => {}", input_shape, out_shape);
                }

                (LogicalOpType::ReshapeOnly, vec![out_shape])
            }
            "ConstantOfShape" => {
                let output_shape =
                    self.find_stored_shape(node.input[0].as_str())
                        .ok_or_else(|| {
                            anyhow!(
                                "failed to find shape {} for ConstantOfShapea",
                                node.input[0]
                            )
                        })?;

                if output_shape.ndims() == 1 {
                    match node.attribute.iter().find(|attr| attr.name() == "value") {
                        Some(attr) if attr.t.data_type() == 7 => {
                            let data = int_slice_from_tensor(&attr.t);
                            let data = std::iter::repeat(data[0])
                                .take(output_shape.numel().unwrap())
                                .collect();
                            self.constants.insert(node.output[0].as_str(), data);
                        }
                        _ => {}
                    }
                }

                let constant = match node.attribute.iter().find(|attr| attr.name() == "value") {
                    Some(attr) if attr.t.data_type() == 1 => {
                        let data = float_slice_from_tensor(&attr.t);
                        data[0]
                    }
                    _ => 0.,
                };

                (
                    LogicalOpType::ConstantOfShape { constant },
                    vec![output_shape],
                )
            }
            "Constant" => {
                let mut shape = None;
                for attr in &node.attribute {
                    match attr.name() {
                        "value_ints" => shape = Some(Shape::from(&[attr.ints.len() as _])),
                        "value_floats" => shape = Some(Shape::from(&[attr.floats.len() as _])),
                        "value" => {
                            shape = Some(Shape::from(&attr.t.dims));

                            assert!(attr.t.has_raw_data());
                            if attr.t.data_type() == 7 {
                                self.constants.insert(
                                    node.output[0].as_str(),
                                    int_slice_from_tensor(&attr.t).to_vec(),
                                );
                            }
                        }
                        "value_float" | "value_int" => shape = Some(Shape::empty()),
                        other => {
                            unimplemented!("{other}");
                        }
                    }

                    if shape.is_some() {
                        break;
                    }
                }

                let Some(attr) = node.attribute.first() else {
                    bail!("attribute not provided for Constant {}", node.name());
                };
                let mut i = [0];
                let mut f = [0.];
                let data: &[u8] = match attr.name() {
                    "value" => attr.t.raw_data(),
                    "value_int" => {
                        i[0] = attr.i();
                        bytemuck::cast_slice(&i)
                    }
                    "value_ints" => bytemuck::cast_slice(&attr.ints),
                    "value_float" => {
                        f[0] = attr.f();
                        bytemuck::cast_slice(&f)
                    }
                    "value_floats" => bytemuck::cast_slice(&attr.floats),
                    _ => bail!("unsupported Constant type '{}'", attr.name()),
                };

                (
                    LogicalOpType::Constant {
                        constant: data.to_vec(),
                    },
                    vec![shape.ok_or_else(|| anyhow!("could not infer shape of constant"))?],
                )
            }
            "Shape" => {
                assert!(node.input.len() == 1);
                let input_shape = &self.shapes[node.input[0].as_str()];

                self.known_shapes
                    .insert(node.output[0].as_str(), input_shape.clone());

                if input_shape.is_concrete() {
                    self.constants
                        .insert(node.output[0].as_str(), input_shape.as_ints()?);
                }

                (
                    LogicalOpType::Shape,
                    vec![Shape::from(&[input_shape.ndims() as _])],
                )
            }
            "GlobalAveragePool" => {
                let input_shape = &self.shapes[node.input[0].as_str()];
                let output_shape = Shape::from(&[
                    input_shape.concrete_size(0)? as i64,
                    input_shape.concrete_size(1)? as i64,
                    1,
                    1,
                ]);
                (LogicalOpType::GlobalAveragePool, vec![output_shape])
            }
            "BatchNormalization" => (
                LogicalOpType::BatchNormalization {
                    epsilon: get_attr_float(node, "epsilon").unwrap_or(1e-4),
                },
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "InstanceNormalization" => (
                LogicalOpType::InstanceNormalization {
                    epsilon: get_attr_float(node, "epsilon").unwrap_or(1e-4),
                },
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Cast" => {
                let target_type = DataType::from_int(
                    get_attr_int(node, "to")
                        .ok_or_else(|| anyhow!("could not find attribute 'to'"))
                        .with_context(|| {
                            anyhow!("when building {}[{}]", node.name(), node.op_type())
                        })? as i32,
                )?;
                (
                    LogicalOpType::Cast { target_type },
                    vec![self.shapes[node.input[0].as_str()].clone()],
                )
            }
            "LeakyRelu" => (
                LogicalOpType::LeakyRelu {
                    alpha: get_attr_float(node, "alpha").unwrap_or(0.01),
                },
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Relu" => (
                LogicalOpType::Relu,
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Erf" => (
                LogicalOpType::Erf,
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Sqrt" => (
                LogicalOpType::Sqrt,
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Sigmoid" => (
                LogicalOpType::Sigmoid,
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Softmax" => (
                LogicalOpType::Softmax {
                    axis: get_attr_int(node, "axis").unwrap_or(-1),
                },
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Tanh" => (
                LogicalOpType::Tanh,
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Sin" => (
                LogicalOpType::Sin,
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Cos" => (
                LogicalOpType::Cos,
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Exp" => (
                LogicalOpType::Exp,
                vec![self.shapes[node.input[0].as_str()].clone()],
            ),
            "Slice" => {
                let x = &self.shapes[node.input[0].as_str()];
                let r = x.ndims() as i64;

                let range: Vec<i64> = (0..x.ndims() as i64).collect();
                let axes: Vec<i64> = node
                    .input
                    .get(3)
                    .and_then(|input| self.find_constant(input.as_str()))
                    .or_else(|| get_attr_ints(node, "axes"))
                    .unwrap_or(&range)
                    .iter()
                    .map(|ax| if *ax < 0 { *ax + r } else { *ax })
                    .collect();

                let starts = node
                    .input
                    .get(1)
                    .and_then(|input| self.find_constant(input.as_str()))
                    .or_else(|| get_attr_ints(node, "starts"))
                    .ok_or_else(|| anyhow!("could not find starts {}", node.input[1]))?;
                let ends = node
                    .input
                    .get(2)
                    .and_then(|input| self.find_constant(input.as_str()))
                    .or_else(|| get_attr_ints(node, "ends"))
                    .ok_or_else(|| anyhow!("could not find ends {}", node.input[2]))?;

                let default_steps = std::iter::repeat(1).take(axes.len()).collect::<Vec<i64>>();
                let steps = node
                    .input
                    .get(4)
                    .and_then(|input| self.find_constant(input.as_str()))
                    .unwrap_or(&default_steps);
                let mut out = Shape::empty();

                for dim in 0isize..x.ndims() as isize {
                    if let Some(i) = axes.iter().enumerate().find_map(|(i, ax_dim)| {
                        if &(dim as i64) == ax_dim {
                            Some(i)
                        } else {
                            None
                        }
                    }) {
                        let start = starts[i];
                        let end = ends[i];
                        let step = steps[i];

                        let elems_along_dim =
                            (end.clamp(0, x.concrete_size(dim)? as i64) - start) / step;
                        out.append_dim(
                            crate::shape::Dimension::Concrete(elems_along_dim as usize).into(),
                        );
                    } else {
                        out.append_dim(x.size(dim).clone());
                    }
                }

                (
                    LogicalOpType::Slice {
                        axes,
                        starts: starts.to_vec(),
                        ends: ends.to_vec(),
                        steps: steps.to_vec(),
                    },
                    vec![out],
                )
            }
            "Where" => {
                let a = &self.shapes[node.input[0].as_str()];
                let b = &self.shapes[node.input[1].as_str()];
                let c = &self.shapes[node.input[2].as_str()];

                if let (Some(a), Some(b), Some(c)) = (
                    self.find_constant(node.input[0].as_str()),
                    self.find_constant(node.input[1].as_str()),
                    self.find_constant(node.input[2].as_str()),
                ) {
                    let d = a
                        .iter()
                        .zip(std::iter::zip(b, c))
                        .map(|(a, (b, c))| if *a == 1 { *b } else { *c })
                        .collect();
                    self.constants.insert(node.output[0].as_str(), d);
                }

                if a != b || b != c {
                    bail!("invalid Where({}, {}, {})", a, b, c);
                }

                (LogicalOpType::Where, vec![a.clone()])
            }
            "Equal" => {
                let a = &self.shapes[node.input[0].as_str()];
                let b = &self.shapes[node.input[1].as_str()];
                if a != b {
                    bail!("invalid Equal {} == {}", a, b);
                }

                if let (Some(a), Some(b)) = (
                    self.find_constant(node.input[0].as_str()),
                    self.find_constant(node.input[1].as_str()),
                ) {
                    let out = a.iter().zip(b).map(|(a, b)| (a == b) as i64).collect();
                    self.constants.insert(node.output[0].as_str(), out);
                }

                (LogicalOpType::Equal, vec![a.clone()])
            }
            "ReduceMean" => {
                let axes =
                    get_attr_ints(node, "axes").ok_or_else(|| anyhow!("failed to find axes"))?;
                let keepdims = get_attr_int(node, "keepdims").unwrap_or(1);

                match keepdims {
                    1 => {}
                    0 => bail!("keepdims = 1 is only supported"),
                    other => bail!("invalid '{other}' value for keepdims"),
                }

                let data = &self.shapes[node.input[0].as_str()];

                let mut out = data.clone();
                for dim in axes {
                    if keepdims == 1 {
                        out.set_dim(*dim as isize, crate::shape::Dimension::Concrete(1));
                    }
                }

                (LogicalOpType::ReduceMean, vec![out])
            }
            "Identity" => {
                let input_shape = self.shapes[node.input[0].as_str()].clone();

                if let Some(val) = self.constants.get(node.input[0].as_str()) {
                    self.constants.insert(node.input[0].as_str(), val.clone());
                }

                if let Some(val) = self.known_shapes.get(node.input[0].as_str()) {
                    self.known_shapes
                        .insert(node.output[0].as_str(), val.clone());
                }

                if input_shape.ndims() == 1 {
                    if let Ok(slice_to_insert) = self
                        .graph
                        .initializer
                        .iter()
                        .find(|init| init.data_type() == 7 && init.name() == node.input[0])
                        .map(int_slice_from_tensor)
                        .ok_or_else(|| anyhow!("failed to find shape for tensor {}", node.input[0]))
                    {
                        self.constants
                            .insert(node.output[0].as_str(), slice_to_insert.to_vec());
                    }
                }

                if let Some(alias) = self.aliases.get(node.input[0].as_str()) {
                    self.aliases.insert(&node.output[0], alias);
                } else {
                    self.aliases.insert(&node.output[0], &node.input[0]);
                }
                (LogicalOpType::ReshapeOnly, vec![input_shape])
            }
            other => unimplemented!("node type {}", other),
        };
        Ok(out)
    }

    fn find_constant(&self, node: &str) -> Option<&[i64]> {
        match self.constants.get(node) {
            Some(s) => Some(s),
            None => self
                .graph
                .initializer
                .iter()
                .find(|init| init.data_type() == 7 && init.name() == node)
                .map(int_slice_from_tensor),
        }
    }

    fn find_stored_shape(&self, node: &str) -> Option<Shape> {
        match self.known_shapes.get(node) {
            Some(s) => Some(s.clone()),
            None => self.find_constant(node).map(Shape::from),
        }
    }

    pub(super) fn get_shape(&self, node: &str) -> &Shape {
        &self.shapes[node]
    }

    pub(super) fn init(&mut self, node: &'a str, shape: Shape) {
        self.shapes.insert(node, shape);
    }
}
