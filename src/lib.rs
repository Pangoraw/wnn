use tensor::CPUTensor;

use std::{collections::HashMap, ops::Sub, path::PathBuf};

use anyhow::{anyhow, bail, Context, Result};

use crate::{
    compiler::{effective_inputs, is_reshape_op, is_untracked_op},
    gpu::{Op, TensorDesc},
    shape::Shape,
    tensor::DataType,
};

mod analyzer;
mod compiler;
mod gpu;
pub mod npy;
pub mod onnx;
mod shape;
mod tensor;
mod utils;

#[derive(Debug)]
pub enum InitMode<'a> {
    Ones,
    Range,
    File(&'a str),
}

pub type EvalOutput<'a> = HashMap<&'a str, CPUTensor>;

pub fn eval_graph<'a>(
    graph: &'a onnx::GraphProto,
    init: InitMode<'_>,
    dump_folder: Option<PathBuf>,
) -> Result<EvalOutput<'a>> {
    // Concretize common dimensions, TODO: pass as arg
    let dim_mappings = HashMap::from_iter([
        ("N", shape::Dimension::Concrete(1)),
        ("batch", shape::Dimension::Concrete(1)),
        ("channels", shape::Dimension::Concrete(4)),
        ("height", shape::Dimension::Concrete(64)),
        ("width", shape::Dimension::Concrete(64)),
    ]);

    let descs = analyzer::analyze_graph(&graph, &dim_mappings)?;

    let mut s = 0;
    for output in &graph.output {
        let desc = &descs[output.name()];
        let computed_shape = &desc.shape;
        let computed_type = &desc.dtype;
        println!("{}{}::{}", output.name(), computed_shape, computed_type);
        if let Some(shape) = &output.type_.tensor_type().shape.0 {
            let real_shape = Shape::from_tensor_shape(shape);
            if computed_shape != &real_shape && real_shape.is_concrete() {
                bail!(
                    "{}: computed {}, stored {}",
                    output.name(),
                    computed_shape,
                    real_shape
                );
            } else if real_shape.is_concrete() {
                s += 1;
            }
        }
    }

    // Validation
    for val in &graph.value_info {
        if let Some(shape) = &val.type_.tensor_type().shape.0 {
            let desc = &descs[val.name()];
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

    let enable_f16 = descs
        .values()
        .any(|desc: &TensorDesc| matches!(desc.dtype, DataType::F16));
    let max_buffer_size = descs.values().map(|desc| desc.size_of() as u32).max();
    let mut runner = pollster::block_on(gpu::Runner::new(max_buffer_size, enable_f16))?;

    for init in &graph.initializer {
        let desc = &descs[init.name()];
        runner
            .add_init(init, desc.clone())
            .with_context(|| anyhow!("failed to create buffer for node {}", init.name()))?;
    }

    for input in &graph.input {
        let desc = &descs[input.name()];
        let shape = &desc.shape;
        let dtype = &desc.dtype;

        log::debug!("using input \"{:?}\"", &init);

        let numel = shape.numel().unwrap();
        let floats: Vec<u8> = match (&init, dtype) {
            (InitMode::Ones, DataType::F32) => bytemuck::cast_slice(
                &std::iter::repeat([1.0])
                    .flatten()
                    .take(numel)
                    .collect::<Vec<f32>>(),
            )
            .to_vec(),
            (InitMode::Ones, DataType::F64) => bytemuck::cast_slice(
                &std::iter::repeat([1.0])
                    .flatten()
                    .take(numel)
                    .collect::<Vec<f64>>(),
            )
            .to_vec(),
            (InitMode::Ones, DataType::I64) => bytemuck::cast_slice(
                &std::iter::repeat([1])
                    .flatten()
                    .take(numel)
                    .collect::<Vec<i64>>(),
            )
            .to_vec(),
            (InitMode::Range, DataType::F32) => bytemuck::cast_slice(
                &(0..numel)
                    .map(|i| i as f32)
                    .take(numel)
                    .collect::<Vec<f32>>(),
            )
            .to_vec(),
            (InitMode::Range, DataType::F64) => bytemuck::cast_slice(
                &(0..numel)
                    .map(|i| i as f64)
                    .take(numel)
                    .collect::<Vec<f64>>(),
            )
            .to_vec(),
            (InitMode::File(path), _) if path.ends_with(".npy") => {
                let (read_shape, data) = npy::read_from_file(path)
                    .with_context(|| anyhow!("when open file {}", path))?;
                if shape != &read_shape.shape {
                    bail!(
                        "invalid shape from input file {}, expected {shape}",
                        read_shape.shape
                    );
                }
                data
            }
            (InitMode::File(path), DataType::F32)
                if path.ends_with(".jpeg") || path.ends_with(".jpg") =>
            {
                let img = image::open(path)?;
                let buf = img.to_rgb32f();
                bytemuck::cast_slice(&buf).to_vec()
            }
            (init, dtype) => bail!("invalid initialization '{:?}' for type {dtype}", init),
        };

        // Some inputs can also be present in initializers, skip those
        if graph
            .initializer
            .iter()
            .any(|init| init.name() == input.name())
        {
            continue;
        }

        runner.add_node_with_init(input.name(), desc.clone(), bytemuck::cast_slice(&floats))?;
    }

    for node in graph
        .node
        .iter()
        .filter(|node| node.op_type() == "Constant")
    {
        let output = &node.output[0];
        let desc = &descs[output.as_str()];

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

        runner.add_node_with_init(output.as_str(), desc.clone(), data)?;
    }

    for output in &graph.output {
        let desc = &descs[output.name()];
        runner.add_node(output.name(), desc.clone(), true)?;
    }

    runner
        .allocate_tensors(&graph.node, &descs, dump_folder.is_some())
        .with_context(|| anyhow!("when allocating nodes"))?;

    log::info!(
        "total_alloc_size = {}",
        human_bytes::human_bytes(runner.total_allocated_size() as f64)
    );
    // return Ok(());

    log::info!("building ops");

    let ops = graph
        .node
        .iter()
        .filter(|node| !is_reshape_op(node.op_type()) && !is_untracked_op(node.op_type()))
        .map(|node| {
            Op::new(
                &runner.device,
                node.input
                    .iter()
                    .take(effective_inputs(node))
                    .map(|input| runner.get_storage(input))
                    .collect::<anyhow::Result<Vec<&gpu::TensorStorage>>>()?,
                node.output
                    .iter()
                    .map(|output| runner.get_storage(output))
                    .collect::<anyhow::Result<Vec<&gpu::TensorStorage>>>()?,
                node,
                &descs,
            )
        })
        .collect::<anyhow::Result<Vec<Op>>>()?;

    log::info!("submitting ops");
    let time = std::time::Instant::now();

    // We submit things in chunks as it turns out submitting 500+ shaders at once
    // is not really appreciated by the GPU :((
    let chunk_size = 10;

    for chunk in ops.chunks(chunk_size) {
        let mut encoder = runner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Command Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            for op in chunk {
                op.run(&mut compute_pass);
            }
        }
        runner.queue.submit(std::iter::once(encoder.finish()));
    }

    let outputs = if let Some(folder) = dump_folder {
        if !folder.exists() {
            std::fs::create_dir(folder)?;
        }

        descs
            .iter()
            .map(|(inter, desc)| -> Result<(&str, CPUTensor)> {
                let tensor = runner.get_storage(inter)?;

                let tensor_bytes = tensor.read_bytes(&runner.device, &runner.queue);
                // let tensor_vec: &[f32] = bytemuck::cast_slice(&tensor_bytes);
                // npy::save_to_file(&format!("activations/{inter}.npy"), tensor_vec, &desc.shape)?;
                Ok((inter, CPUTensor::new(desc.clone(), &tensor_bytes)))
            })
            .collect::<Result<Vec<(&str, CPUTensor)>>>()?
    } else {
        graph
            .output
            .iter()
            .map(|output| -> Result<(&str, CPUTensor)> {
                let tensor = runner.get_storage(output.name())?;
                let tensor_bytes = tensor.read_bytes(&runner.device, &runner.queue);
                let desc = &descs[output.name()];

                let elapsed = std::time::Instant::now().sub(time);
                log::info!("run done ({:?})", elapsed);

                // let tensor_vec: &[f32] = bytemuck::cast_slice(&tensor_bytes);

                // let out_shape = &descs[output.name()].shape;
                // let filename = format!("activations/{}.npy", output.name());
                // log::info!("saving to file {filename}");

                // npy::save_to_file(&filename, tensor_vec, out_shape)
                //     .with_context(|| anyhow!("failed to save to file {filename}"))?;

                Ok((
                    output.name(),
                    CPUTensor::new(desc.clone(), &tensor_bytes),
                ))
            })
            .collect::<Result<Vec<(&str, CPUTensor)>>>()?
    };
    let outputs = HashMap::from_iter(outputs);
    log::info!("done!");

    Ok(outputs)
}
