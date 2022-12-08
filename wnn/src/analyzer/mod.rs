use std::collections::HashMap;

use anyhow::{anyhow, Context};

use crate::{
    onnx,
    shape::{self, Shape},
    tensor::{DataType, TensorDesc},
};

mod shape_inference;
mod type_inference;

pub(crate) type BufferHandle = usize;
pub(crate) struct LogicalBuffer<'a> {
    name: &'a str,
    desc: TensorDesc,
}

#[derive(Debug)]
pub(crate) enum ResizeCoordinateTransformationMode {
    Asymmetric,
}

#[derive(Debug)]
pub(crate) enum PoolType {
    Max,
    Conv,
}

#[derive(Debug)]
pub(crate) enum LogicalOpType {
    Sqrt,
    Tanh,
    Erf,
    Gemm {
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
    },
    BatchNormalization {
        epsilon: f32,
    },
    InstanceNormalization {
        epsilon: f32,
    },
    Resize {
        transformation_mode: ResizeCoordinateTransformationMode,
    },
    Softmax {
        axis: i64,
    },
    Gather {
        axis: i64,
    },
    Slice {
        axes: Vec<i64>,
        starts: Vec<i64>,
        ends: Vec<i64>,
        steps: Vec<i64>,
    },
    Constant,
    ConstantOfShape {
        constant: f32,
    },
    Concat {
        axis: i64,
    },
    Cast {
        target_type: DataType,
    },
    Cos,
    Sin,
    Relu,
    LeakyRelu {
        alpha: f32,
    },
    Sigmoid,
    Pool {
        ptype: PoolType,
        group: i64,
        dilations: [i64; 2],
        k_strides: [i64; 2],
        pads: [i64; 4],
        kernel_shape: [i64; 2],
    },
    Transpose {
        perm: Vec<i64>,
    },
    ReduceMean,
    Where,
    Equal,
    Mul,
    Add,
    Div,
    Sub,
    Pow,
    Expand,
    GlobalAveragePool,

    Shape,
    ReshapeOnly,
}

impl std::fmt::Display for LogicalOpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use LogicalOpType::*;

        f.write_str(match self {
            Sqrt => "Sqrt",
            Tanh => "Tanh",
            Erf => "Erf",
            Gemm { .. } => "Gemm",
            BatchNormalization { .. } => "BatchNormalization",
            InstanceNormalization { .. } => "InstanceNormalization",
            Resize { .. } => "Resize",
            Softmax { .. } => "Softmax",
            Gather { .. } => "Gather",
            Slice { .. } => "Slice",
            Constant => "Constant",
            ConstantOfShape { .. } => "ConstantOfShape",
            Concat { .. } => "Concat",
            Cast { .. } => "Cast",
            Cos => "Cos",
            Sin => "Sin",
            Relu => "Relu",
            LeakyRelu { .. } => "LeakyRelu",
            Sigmoid => "Sigmoid",
            Pool {
                ptype: PoolType::Max,
                ..
            } => "Maxpool",
            Pool {
                ptype: PoolType::Conv,
                ..
            } => "Conv",
            Transpose { .. } => "Transpose",
            ReduceMean => "ReduceMean",
            Where => "Where",
            Equal => "Equal",
            Mul => "Mul",
            Add => "Add",
            Div => "Div",
            Sub => "Sub",
            Pow => "Pow",
            Expand => "Expand",
            GlobalAveragePool => "GlobalAveragePool",

            Shape => "Shape",
            ReshapeOnly => "ReshapeOnly",
        })
    }
}

pub(crate) struct LogicalOp<'a> {
    name: Option<&'a str>,
    pub(crate) inputs: Vec<BufferHandle>,
    pub(crate) outputs: Vec<BufferHandle>,
    op_type: LogicalOpType,
}

impl<'a> LogicalOp<'a> {
    pub(crate) fn name(&self) -> &'a str {
        self.name.unwrap_or("unknown")
    }

    pub(crate) fn op_type(&'a self) -> &'a LogicalOpType {
        &self.op_type
    }
}

pub(crate) struct LogicalGraph<'a> {
    pub(crate) buffers: Vec<LogicalBuffer<'a>>,
    pub(crate) ops: Vec<LogicalOp<'a>>,
}

impl<'a> LogicalGraph<'a> {
    pub(crate) fn new(
        graph: &'a onnx::GraphProto,
        dim_mappings: &HashMap<&str, shape::Dimension>,
    ) -> anyhow::Result<LogicalGraph<'a>> {
        let mut logical_graph = Self {
            buffers: Vec::new(),
            ops: Vec::with_capacity(graph.node.len()),
        };
        logical_graph.infer_graph(graph, dim_mappings)?;
        Ok(logical_graph)
    }

    fn add_buffer(&mut self, name: &'a str, desc: TensorDesc) -> BufferHandle {
        let handle = self.buffers.len();
        self.buffers.push(LogicalBuffer { name, desc });
        handle
    }

    pub(crate) fn find_buffer_handle(&self, name: &str) -> Option<BufferHandle> {
        self.buffers.iter().enumerate().find_map(|(index, buffer)| {
            if buffer.name == name {
                Some(index)
            } else {
                None
            }
        })
    }

    pub(crate) fn get_desc(&self, handle: BufferHandle) -> &TensorDesc {
        &self.buffers[handle].desc
    }

    pub(crate) fn get_desc_name(&self, name: &str) -> &TensorDesc {
        match self.find_buffer_handle(name) {
            Some(i) => self.get_desc(i),
            None => panic!("cannot find handle for {name}"),
        }
    }

    pub(crate) fn max_buffer_size(&self) -> Option<u32> {
        self.buffers
            .iter()
            .map(|buffer| buffer.desc.size_of() as u32)
            .max()
    }

    fn infer_graph(
        &mut self,
        graph: &'a onnx::GraphProto,
        dim_mappings: &HashMap<&str, shape::Dimension>,
    ) -> anyhow::Result<()> {
        let mut shape_inferer = shape_inference::ShapeInferer::new(graph);
        let mut dtype_inferer = type_inference::TypeInferer::new();

        for init in &graph.initializer {
            let shape = Shape::from(&init.dims);
            let dtype = DataType::from_int(init.data_type())?;
            dtype_inferer.init(init.name(), dtype.clone());
            shape_inferer.init(init.name(), shape.clone());
            self.add_buffer(init.name(), TensorDesc::new(shape, dtype));
        }

        println!("==== INPUT");
        for input in &graph.input {
            let dtype = DataType::from_int(input.type_.tensor_type().elem_type())?;
            let shape = {
                let tensor_shape = input.type_.tensor_type().shape.as_ref().ok_or_else(|| {
                    anyhow!("failed to get tensor shape for input {}", input.name())
                })?;
                Shape::from_tensor_shape_with_maps(tensor_shape, dim_mappings)
            };
            println!("{}{}::{}", input.name(), &shape, &dtype);
            dtype_inferer.init(input.name(), dtype.clone());
            shape_inferer.init(input.name(), shape.clone());
            self.add_buffer(input.name(), TensorDesc::new(shape, dtype));
        }
        println!("======");

        let mut intermediaries = Vec::new();
        for node in &graph.node {
            let out_types = dtype_inferer.infer_node(node).with_context(|| {
                format!(
                    "processing shapes for node {}[{}]({})",
                    node.name(),
                    node.op_type(),
                    node.input
                        .iter()
                        .map(|input| format!("{}{}", input, shape_inferer.get_shape(input)))
                        .collect::<Vec<String>>()
                        .join(", "),
                )
            })?;
            let (op_type, out_shapes) = shape_inferer.infer_node(node).with_context(|| {
                format!(
                    "processing shapes for node {}[{}]({})",
                    node.name(),
                    node.op_type(),
                    node.input
                        .iter()
                        .map(|input| format!("{}{}", input, shape_inferer.get_shape(input)))
                        .collect::<Vec<String>>()
                        .join(", "),
                )
            })?;

            // TODO: maybe hashmap ?
            let inputs: Vec<BufferHandle> = node
                .input
                .iter()
                .take(effective_inputs_(&op_type, node.input.len()))
                .map(|input| {
                    self.find_buffer_handle(input)
                        .ok_or_else(|| anyhow!("failed to find node {input}"))
                })
                .collect::<anyhow::Result<Vec<BufferHandle>>>()?;

            let mut is_longint_op = false;
            let outputs: Vec<BufferHandle> = node
                .output
                .iter()
                .zip(std::iter::zip(out_types, out_shapes))
                .map(|(out, (out_type, out_shape))| {
                    if matches!(out_type, DataType::I64) {
                        is_longint_op = true;
                    }
                    intermediaries.push(out);
                    shape_inferer.init(out, out_shape.clone());
                    dtype_inferer.init(out, out_type.clone());
                    self.add_buffer(out, TensorDesc::new(out_shape, out_type))
                })
                .collect();

            let op = LogicalOp {
                name: Some(node.name()),
                inputs,
                outputs,
                op_type,
            };

            if &std::env::var_os("DUMP_INFERENCE")
                .map(|s| s.to_str().unwrap().to_owned())
                .unwrap_or_else(|| String::from("1"))
                == "1"
            {
                for (i, out) in op.outputs.iter().enumerate() {
                    let name = self.buffers[*out].name;
                    print!(
                        "{}[%{}]{}::{}",
                        name,
                        out,
                        shape_inferer.get_shape(name),
                        dtype_inferer.get_type(name)
                    );
                    if i < op.outputs.len() - 1 {
                        print!(", ");
                    }
                }
                print!("\t= {}[{}](", op.name(), op.op_type());
                for (i, input) in op.inputs.iter().enumerate() {
                    let name = self.buffers[*input].name;
                    if name.is_empty() {
                        print!("None");
                    } else {
                        print!(
                            "{}[%{}]{}",
                            name,
                            input,
                            shape_inferer.get_shape(name)
                        );
                    }
                    if i < op.inputs.len() - 1 {
                        print!(", ");
                    }
                }
                println!(")");
            }

            // Don't include integer ops in the logical graph
            // wgsl don't support i64.
            if !is_longint_op {
                self.ops.push(op);
            }
        }

        Ok(())
    }

    pub(crate) fn names(&'a self) -> Vec<&'a str> {
        self.buffers.iter().map(|buf| buf.name).collect()
    }
}

fn effective_inputs_(op_type: &LogicalOpType, default: usize) -> usize {
    match op_type {
        LogicalOpType::ConstantOfShape { .. } => 0,
        LogicalOpType::Resize { .. } | LogicalOpType::ReshapeOnly | LogicalOpType::Slice { .. } => {
            1
        } // These ops are tracked statically so we remove tensor "params"
        _ => default,
    }
}

pub(crate) fn effective_inputs(op: &LogicalOp) -> usize {
    effective_inputs_(&op.op_type, op.inputs.len())
}
