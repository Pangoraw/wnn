use std::collections::HashMap;

use anyhow::{anyhow, bail};

use crate::onnx;
use crate::shape::Shape;
use crate::utils::*;

pub(crate) struct ShapeInferer<'a> {
    shapes: HashMap<&'a str, Shape>,
    constants: HashMap<&'a str, Shape>,
    aliases: HashMap<&'a str, &'a str>,
    graph: &'a onnx::GraphProto,
}

impl<'a> ShapeInferer<'a> {
    pub(crate) fn new(graph: &'a onnx::GraphProto) -> Self {
        Self {
            shapes: HashMap::new(),
            constants: HashMap::new(),
            aliases: HashMap::new(),
            graph,
        }
    }

    pub(crate) fn infer_node(&mut self, node: &'a onnx::NodeProto) -> anyhow::Result<Vec<Shape>> {
        let out = match node.op_type() {
            op @ ("Gemm" | "MatMul") => {
                let transpose_a = if op == "Gemm" {
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
                } else {
                    0
                };
                let transpose_b = if op == "Gemm" {
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
                } else {
                    0
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

                if transpose_a == 1 {
                    a.transpose(-1, -2)
                };
                if transpose_b == 1 {
                    b.transpose(-1, -2)
                };

                if let Some(c) = c {
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
                vec![out_dim]
            }
            "Mul" | "Add" | "Div" | "Sub" => {
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

                vec![out_shape]
            }
            "Expand" => {
                let mut out = self.shapes[node.input[0].as_str()].clone();
                let shape = self
                    .find_stored_shape(&node.input[1])
                    .ok_or_else(|| anyhow!("failed to find shape {}", node.input[1]))?;

                out.broadcast(&shape)?;
                vec![out]
            }
            "Transpose" => {
                let perm = get_attr_ints(node, "perm")
                    .ok_or_else(|| anyhow!("invalid transpose: perm not found"))?;
                assert!(node.input.len() == 1);
                let mut shape = self.shapes[node.input[0].as_str()].clone();
                shape.permute(perm);

                vec![shape]
            }
            "MaxPool" => {
                let kernel_shape = get_attr_ints(node, "kernel_shape")
                    .ok_or_else(|| anyhow!("kernel_shape not provided"))?;
                let mut input_shape = self.shapes[node.input[0].as_str()].clone();
                let factors = vec![
                    1.,
                    1.,
                    1. / kernel_shape[0] as f32,
                    1. / kernel_shape[1] as f32,
                ];
                input_shape.scale(&factors)?;
                vec![input_shape]
            }
            "Flatten" => {
                let input_shape = &self.shapes[node.input[0].as_str()];
                let mut output_shape = Shape::empty();
                output_shape.append_dim(input_shape.size(0).clone());
                output_shape.append_dim(crate::shape::Dimension::Rest);
                vec![output_shape.map_and_rest(input_shape)]
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

                vec![out_shape]
            }
            "Unsqueeze" => {
                let mut out_shape = self.shapes[node.input[0].as_str()].clone();
                let axes = self
                    .find_stored_shape(&node.input[1])
                    .ok_or_else(|| anyhow!("failed to find shape for tensor {}", node.input[1]))?;

                for dim in 0isize..axes.ndims() as isize {
                    out_shape.unsqueeze(axes.as_int(dim)? as usize);
                }

                vec![out_shape]
            }
            "Gather" => {
                let axis = get_attr_int(node, "axis").unwrap_or(0);
                let data = &self.shapes[node.input[0].as_str()];
                let indices = &self.shapes[node.input[1].as_str()];

                let out_shape = if indices.is_scalar() {
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
                vec![out_shape]
            }
            "Conv" => {
                let input_shapes = node
                    .input
                    .iter()
                    .map(|input| {
                        self.shapes
                            .get(input.as_str())
                            .ok_or_else(|| anyhow!("failed to get shape for input {}", input))
                    })
                    .collect::<anyhow::Result<Vec<&Shape>>>()?;
                let pads =
                    get_attr_ints(node, "pads").ok_or_else(|| anyhow!("could not find pads"))?;
                let strides = get_attr_ints(node, "strides")
                    .ok_or_else(|| anyhow!("could not find strides"))?;
                let x = input_shapes[0];
                let w = input_shapes[1];

                if x.ndims() != 4 || w.ndims() != 4 {
                    bail!("invalid convolution {} o {}", x, w);
                }
                if pads.len() != 4 {
                    bail!("invalid pads {:?}", pads);
                }

                let mut out_shape = Shape::empty();
                out_shape.append_dim(x.size(0).clone()); // B
                out_shape.append_dim(w.size(0).clone()); // C
                out_shape.append_dim(crate::shape::Dimension::Concrete(
                    (x.concrete_size(2)? - 2 * (w.concrete_size(2)? / 2)
                        + pads[0] as usize
                        + pads[2] as usize)
                        / *strides.first().unwrap_or(&1) as usize,
                )); // H
                out_shape.append_dim(crate::shape::Dimension::Concrete(
                    (x.concrete_size(3)? - 2 * (w.concrete_size(3)? / 2)
                        + pads[1] as usize
                        + pads[3] as usize)
                        / *strides.get(1).unwrap_or(&1) as usize,
                )); // W

                vec![out_shape]
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

                vec![input_shape]
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

                vec![out_shape]
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
                vec![output_shape]
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
                                    Shape::from(int_slice_from_tensor(&attr.t)),
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

                vec![shape.ok_or_else(|| anyhow!("could not infer shape of constant"))?]
            }
            "Shape" => {
                assert!(node.input.len() == 1);
                let input_shape = &self.shapes[node.input[0].as_str()];
                self.constants
                    .insert(node.output[0].as_str(), input_shape.clone());
                vec![Shape::from(&[input_shape.ndims() as _])]
            }
            "InstanceNormalization" | "Cast" | "Relu" | "Sigmoid" | "Softmax" | "Tanh" => {
                vec![self.shapes[node.input[0].as_str()].clone()]
            }
            "Where" => {
                let a = &self.shapes[node.input[0].as_str()];
                let b = &self.shapes[node.input[1].as_str()];
                let c = &self.shapes[node.input[2].as_str()];
                if a != b || b != c {
                    bail!("invalid Where({}, {}, {})", a, b, c);
                }
                vec![a.clone()]
            }
            "Equal" => {
                let a = &self.shapes[node.input[0].as_str()];
                let b = &self.shapes[node.input[1].as_str()];
                if a != b {
                    bail!("invalid Equal {} == {}", a, b);
                }
                vec![a.clone()]
            }
            "Identity" => {
                let input_shape = self.shapes[node.input[0].as_str()].clone();
                if let Some(val) = self.constants.get(node.input[0].as_str()) {
                    self.constants.insert(node.output[0].as_str(), val.clone());
                } else if input_shape.ndims() == 1 {
                    if let Ok(shape_to_insert) = self
                        .graph
                        .initializer
                        .iter()
                        .find(|init| init.data_type() == 7 && init.name() == node.input[0])
                        .map(|tensor| Shape::from(int_slice_from_tensor(tensor)))
                        .ok_or_else(|| anyhow!("failed to find shape for tensor {}", node.input[0]))
                    {
                        self.constants
                            .insert(node.output[0].as_str(), shape_to_insert);
                    }
                }

                if let Some(alias) = self.aliases.get(node.input[0].as_str()) {
                    self.aliases.insert(&node.output[0], alias);
                } else {
                    self.aliases.insert(&node.output[0], &node.input[0]);
                }
                vec![input_shape]
            }
            other => unimplemented!("node type {}", other),
        };
        Ok(out)
    }

    fn find_stored_shape(&self, node: &str) -> Option<Shape> {
        match self.constants.get(node) {
            Some(s) => Some(s.clone()),
            None => self
                .graph
                .initializer
                .iter()
                .find(|init| init.data_type() == 7 && init.name() == node)
                .map(|tensor| {
                    let shape = Shape::from(int_slice_from_tensor(tensor));
                    assert!(tensor.dims.len() == 1 && shape.ndims() == tensor.dims[0] as usize);
                    shape
                }),
        }
    }

    pub(crate) fn get_shape(&self, node: &str) -> &Shape {
        &self.shapes[node]
    }

    pub(crate) fn init(&mut self, node: &'a str, shape: Shape) {
        self.shapes.insert(node, shape);
    }
}
