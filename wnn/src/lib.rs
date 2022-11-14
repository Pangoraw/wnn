use tensor::CPUTensor;

use std::{
    collections::{HashMap, HashSet},
    ops::Sub,
    path::PathBuf,
};

use anyhow::{anyhow, bail, Context, Result};

use crate::{
    compiler::{effective_inputs, is_reshape_op, is_untracked_op},
    gpu::Op,
    shape::Shape,
    tensor::{DataType, TensorDesc},
};

mod analyzer;
mod compiler;
mod gpu;
pub mod npy;
pub mod onnx;
pub mod shape;
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

pub type EvalOutput<'a> = HashMap<&'a str, CPUTensor>;

pub async fn eval_graph<'a>(
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

    let descs = analyzer::analyze_graph(graph, &dim_mappings)?;

    let mut s = 0;
    for output in &graph.output {
        let desc = &descs[output.name()];
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
    let mut runner = gpu::Runner::new(max_buffer_size, enable_f16).await?;

    for init in &graph.initializer {
        let desc = &descs[init.name()];
        if matches!(desc.dtype, DataType::I64) {
            // Don't allocate tensor of type i64 (it does not exist in wgpu) and
            // therefore must be statically infered (see crate::analyzer).
            continue;
        }

        runner
            .add_init(init, desc.clone())
            .with_context(|| anyhow!("failed to create buffer for node {}", init.name()))?;
    }

    for input in &graph.input {
        // Some inputs can also be present in initializers, skip those
        if graph
            .initializer
            .iter()
            .any(|init| init.name() == input.name())
        {
            continue;
        }

        let desc = &descs[input.name()];
        let shape = &desc.shape;
        let dtype = &desc.dtype;

        log::debug!("using input \"{:?}\"", &init);

        let numel = shape.numel().unwrap();
        let floats: Vec<u8> = match (&init, dtype) {
            (InitMode::Ones, DataType::F32) => {
                bytemuck::cast_slice(&std::iter::repeat(1.0).take(numel).collect::<Vec<f32>>())
                    .to_vec()
            }
            (InitMode::Ones, DataType::F64) => {
                bytemuck::cast_slice(&std::iter::repeat(1.0).take(numel).collect::<Vec<f64>>())
                    .to_vec()
            }
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
            (InitMode::SliceF32(slice), DataType::F32) => bytemuck::cast_slice(slice).to_vec(),
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

        runner.add_node_with_init(input.name(), desc.clone(), &floats)?;
    }

    let mut constants = HashSet::new();
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

        constants.insert(output.as_str());
        runner.add_node_with_init(output.as_str(), desc.clone(), data)?;
    }

    for output in &graph.output {
        let desc = &descs[output.name()];
        runner.add_node(output.name(), desc.clone(), true)?;
    }

    let allow_not_exact_size_buffers = false; // This can decrease the required amount of memory
    runner
        .allocate_tensors(
            &graph.node,
            &descs,
            dump_folder.is_some(),
            allow_not_exact_size_buffers,
        )
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

    #[cfg(not(target_arch = "wasm32"))]
    let time = std::time::Instant::now();

    runner.submit_ops(&ops);

    let outputs = if let Some(folder) = dump_folder {
        if !folder.exists() {
            std::fs::create_dir(folder)?;
        }

        let mut outputs: Vec<(&str, CPUTensor)> = Vec::new();
        for (inter, desc) in descs.iter() {
            if constants.contains(inter)
                || graph.initializer.iter().any(|init| &init.name() == inter)
                || graph.input.iter().any(|input| &input.name() == inter)
            {
                continue;
            }
            let tensor_bytes = runner.read_bytes_from_name(inter).await?;
            outputs.push((inter, CPUTensor::new(desc.clone(), &tensor_bytes)))
        }

        outputs
    } else {
        let mut outputs: Vec<(&str, CPUTensor)> = Vec::new();
        for output in &graph.output {
            let tensor_bytes = runner.read_bytes_from_name(output.name()).await?;
            let desc = &descs[output.name()];

            #[cfg(not(target_arch = "wasm32"))]
            {
                let elapsed = std::time::Instant::now().sub(time);
                log::info!("run done ({:?})", elapsed);
            }

            outputs.push((output.name(), CPUTensor::new(desc.clone(), &tensor_bytes)));
        }
        outputs
    };
    let outputs = HashMap::from_iter(outputs);
    log::info!("done!");

    Ok(outputs)
}
