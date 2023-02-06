use tensor::CPUTensor;

use std::{collections::HashMap, ops::Sub};

use anyhow::{anyhow, bail, Context, Result};

use crate::{
    analyzer::LogicalOpType,
    compiler::is_untracked_op,
    gpu::{BufferType, Op},
    shape::Shape,
    tensor::DataType,
};

mod analyzer;
mod compiler;
mod gpu;
pub mod npy;
pub mod onnx;
pub mod shape;
mod simplifier;
pub mod tensor;
mod utils;

#[derive(Debug)]
pub enum InitMode<'a> {
    Ones,
    Range,
    File(&'a str),
    SliceF32(&'a [f32]),
    SliceI64(&'a [f32]),
}

pub type EvalOutput = HashMap<String, CPUTensor>;

pub struct CompiledModel {
    log_graph: analyzer::LogicalGraph,
    runner: gpu::Runner,
    ops: Vec<Op>,
}

impl CompiledModel {
    pub async fn new(graph: &onnx::GraphProto) -> anyhow::Result<CompiledModel> {
        // Concretize common dimensions, TODO: pass as arg
        let dim_mappings = HashMap::from_iter([
            ("N", shape::Dimension::Concrete(1)),
            ("batch", shape::Dimension::Concrete(1)),
            ("channels", shape::Dimension::Concrete(4)),
            ("height", shape::Dimension::Concrete(64)),
            ("width", shape::Dimension::Concrete(64)),
        ]);

        let log_graph = {
            let mut log_graph = analyzer::LogicalGraph::new(graph, &dim_mappings)?;
            simplifier::simplify(&mut log_graph)?;
            log_graph
        };

        #[cfg(debug_assertions)]
        validate_graph(graph, &log_graph)?;

        let enable_f16 = false;

        // descs
        // .values()
        // .any(|desc: &TensorDesc| matches!(desc.dtype, DataType::F16));
        let max_buffer_size = log_graph.max_buffer_size();
        let mut runner = gpu::Runner::new(max_buffer_size, enable_f16).await?;

        for init in &graph.initializer {
            let desc = log_graph.get_desc_name(init.name());
            if matches!(desc.dtype, DataType::I64) {
                // Don't allocate tensor of type i64 (it does not exist in wgpu) and
                // therefore must be statically infered (see crate::analyzer).
                log::warn!("not instanciating tensor {} of type i64", init.name());
                continue;
            }

            runner
                .add_init(
                    init,
                    &log_graph.find_buffer_handle(init.name()).ok_or_else(|| {
                        anyhow!("could not find buffer handle for {}", init.name())
                    })?,
                    desc.clone(),
                )
                .with_context(|| anyhow!("failed to create buffer for node {}", init.name()))?;
        }

        for input in &log_graph.inputs {
            let desc = log_graph.get_desc(*input);
            runner.add_node(
                input,
                desc.clone(),
                BufferType::Input, // &floats,
            )?;
        }

        for op in &log_graph.ops {
            let LogicalOpType::Constant { constant } = op.op_type() else {
                continue;
            };
            let output = op.outputs[0];
            let desc = log_graph.get_desc(output);

            runner.add_node_with_init(&output, desc.clone(), constant)?;
        }

        for output in &log_graph.outputs {
            let desc = log_graph.get_desc(*output);
            runner.add_node(output, desc.clone(), BufferType::Output)?;
        }

        let force_readable = false;
        let allow_not_exact_size_buffers = true; // This can decrease the required amount of memory
        runner
            .allocate_tensors(&log_graph, force_readable, allow_not_exact_size_buffers)
            .with_context(|| anyhow!("when allocating nodes"))?;

        log::info!(
            "total_alloc_size = {}",
            human_bytes::human_bytes(runner.total_allocated_size() as f64)
        );

        log::info!("building ops");

        let ops = log_graph
            .ops
            .iter()
            .filter(|op| {
                !matches!(op.op_type(), LogicalOpType::ReshapeOnly)
                    && !is_untracked_op(op.op_type())
            })
            .map(|node| {
                Op::new(
                    &runner,
                    node.inputs
                        .iter()
                        // .take(effective_inputs(node))
                        .map(|input| runner.get_storage(input))
                        .collect::<anyhow::Result<Vec<&gpu::TensorStorage>>>()?,
                    node.outputs
                        .iter()
                        .map(|output| runner.get_storage(output))
                        .collect::<anyhow::Result<Vec<&gpu::TensorStorage>>>()?,
                    node,
                    &log_graph,
                )
            })
            .collect::<anyhow::Result<Vec<Op>>>()?;

        Ok(Self {
            runner,
            log_graph,
            ops,
        })
    }

    pub async fn eval_graph(&self, inputs: &[&[u8]]) -> Result<EvalOutput> {
        if inputs.len() != self.log_graph.inputs.len() {
            bail!(
                "invalid inputs provided (expected {} inputs but got {})",
                self.log_graph.inputs.len(),
                inputs.len()
            )
        }

        for (input, input_handle) in inputs.iter().zip(&self.log_graph.inputs) {
            self.runner.write_node(*input_handle, input).await?;
        }

        log::info!("submitting ops");

        #[cfg(not(target_arch = "wasm32"))]
        let time = std::time::Instant::now();

        self.runner.submit_ops(&self.ops);

        let outputs = {
            let mut outputs: Vec<(String, CPUTensor)> = Vec::new();
            for output in &self.log_graph.outputs {
                let tensor_bytes = self.runner.read_bytes(output).await?;
                let desc = self.log_graph.get_desc(*output);

                #[cfg(not(target_arch = "wasm32"))]
                {
                    let elapsed = std::time::Instant::now().sub(time);
                    log::info!("run done ({:?})", elapsed);
                }

                outputs.push((
                    String::from(self.log_graph.find_name(output)),
                    CPUTensor::new(desc.clone(), &tensor_bytes),
                ));
            }
            outputs
        };

        let outputs = HashMap::from_iter(outputs);
        log::info!("done!");

        Ok(outputs)
    }
}

fn validate_graph(
    graph: &onnx::GraphProto,
    log_graph: &analyzer::LogicalGraph,
) -> anyhow::Result<()> {
    let mut s = 0;

    for output in &graph.output {
        let desc = log_graph.get_desc_name(output.name());
        let computed_shape = &desc.shape;
        let computed_type = &desc.dtype;
        println!("{}{}::{}", output.name(), computed_shape, computed_type);
        if let Some(shape) = &output.type_.tensor_type().shape.0 {
            let real_shape = Shape::from_tensor_shape(shape);
            let real_type = DataType::from_int(output.type_.tensor_type().elem_type())?;
            if computed_shape != &real_shape && real_shape.is_concrete() {
                bail!(
                    "{}: computed {}, stored {}",
                    output.name(),
                    computed_shape,
                    real_shape
                );
            } else if computed_type != &real_type {
                bail!(
                    "{}: computed {}, stored {}",
                    output.name(),
                    computed_type,
                    real_type,
                );
            } else if real_shape.is_concrete() {
                s += 1;
            }
        }
    }

    // Validation
    for val in &graph.value_info {
        if let Some(shape) = &val.type_.tensor_type().shape.0 {
            let desc = log_graph.get_desc_name(val.name());
            let computed_shape = &desc.shape;
            let info_shape = Shape::from_tensor_shape(shape);
            let computed_type = &desc.dtype;
            let info_type = DataType::from_int(val.type_.tensor_type().elem_type())?;
            if computed_type != &info_type {
                log::warn!(
                    "{}: computed {:?}, stored {:?}",
                    val.name(),
                    computed_type,
                    info_type
                );
            } else if computed_shape != &info_shape {
                log::warn!(
                    "{}: computed {}, stored {}",
                    val.name(),
                    computed_shape,
                    info_shape
                );
            } else {
                s += 1;
            }
        }
    }
    log::info!("valided {}/{}", s, graph.node.len());

    if s != graph.node.len() {
        log::debug!(
            "not in info = {:?}",
            graph.node.iter().find_map(|node| {
                if !graph.value_info.iter().any(|val| val.name() == node.name()) {
                    Some(node.name())
                } else {
                    None
                }
            })
        );
    }
    Ok(())
}
