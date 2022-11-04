use std::{collections::HashMap, ops::Sub};

use anyhow::{anyhow, Context};
use protobuf::Message;
use structopt::StructOpt;

use crate::{
    compiler::{effective_inputs, is_reshape_op, is_untracked_op},
    gpu::{Op, TensorDesc},
    shape::Shape,
    tensor::DataType,
};

mod analyzer;
mod compiler;
mod gpu;
mod npy;
mod onnx;
mod shape;
mod tensor;
mod utils;

#[derive(StructOpt, Debug)]
struct Args {
    #[structopt(default_value = "./sd-v1-5-onnx/vae_decoder_sim.onnx")]
    input_model: std::path::PathBuf,
    dump_folder: Option<std::path::PathBuf>,

    #[structopt(long, short, default_value = "ones")]
    init: String,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::from_args();

    let dump_folder = args.dump_folder;
    let filename = args.input_model;

    // let filename = "./model.onnx";
    // let filename = "vae_decoder_sim.onnx";
    // let filename = "vae_decoder.onnx";
    // let filename = "unet.onnx";
    // let filename = "./simple_model.onnx";
    // let filename = "/home/paul/Downloads/resnet18-v1-7.onnx";
    // let filename = "/home/pberg/Downloads/resnet18-v1-7.onnx";
    // let filename = "./sd-v1-5-onnx/vae_decoder_sim.onnx";
    // let filename = "/home/pberg/irisa/diffusers/decoder_v1_4_pytorch_1_1.onnx";
    // let filename = "/home/pberg/irisa/diffusers/decoder_v1_4_fp16_pytorch_fixed.onnx";
    // let filename = "/home/pberg/Projects/ONNX.jl/model.onnx";
    // let filename = "/home/pberg/Projects/ONNX.jl/model_sim.onnx";
    // let filename = "unet_sim2.onnx";
    // let filename = "/home/pberg/Downloads/uc_merced_model(2).onnx";

    let mut onnx_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&filename)
        .with_context(|| format!("while opening file {}", filename.display()))?;
    let model = onnx::ModelProto::parse_from_reader(&mut onnx_file)?;

    // Concretize common dimensions
    let dim_mappings = HashMap::from_iter([
        ("N", shape::Dimension::Concrete(1)),
        ("batch", shape::Dimension::Concrete(1)),
        ("channels", shape::Dimension::Concrete(4)),
        ("height", shape::Dimension::Concrete(64)),
        ("width", shape::Dimension::Concrete(64)),
    ]);

    let graph = model.graph;
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
                anyhow::bail!(
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

        let init: &str = &args.init;
        log::info!("using input \"{init}\"");

        let numel = shape.numel().unwrap();
        let floats: Vec<u8> = match (init, dtype) {
            ("ones", DataType::F32) => bytemuck::cast_slice(
                &std::iter::repeat([1.0])
                    .flatten()
                    .take(numel)
                    .collect::<Vec<f32>>(),
            )
            .to_vec(),
            ("ones", DataType::F64) => bytemuck::cast_slice(
                &std::iter::repeat([1.0])
                    .flatten()
                    .take(numel)
                    .collect::<Vec<f64>>(),
            )
            .to_vec(),
            ("arange", DataType::F32) => bytemuck::cast_slice(
                &(0..numel)
                    .map(|i| i as f32)
                    .take(numel)
                    .collect::<Vec<f32>>(),
            )
            .to_vec(),
            ("arange", DataType::F64) => bytemuck::cast_slice(
                &(0..numel)
                    .map(|i| i as f64)
                    .take(numel)
                    .collect::<Vec<f64>>(),
            )
            .to_vec(),
            _ => {
                let (read_shape, data) = npy::read_from_file(init)
                    .with_context(|| anyhow!("when open file {}", init))?;
                if shape != &read_shape.shape {
                    anyhow::bail!(
                        "invalid shape from input file {}, expected {shape}",
                        read_shape.shape
                    );
                }
                data
            }
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

    if let Some(folder) = dump_folder {
        if !folder.exists() {
            std::fs::create_dir(folder)?;
        }

        for (inter, desc) in descs.iter() {
            let tensor = runner.get_storage(inter)?;

            let tensor_bytes = tensor.read_bytes(&runner.device, &runner.queue);
            let tensor_vec: &[f32] = bytemuck::cast_slice(&tensor_bytes);

            npy::save_to_file(&format!("activations/{inter}.npy"), tensor_vec, &desc.shape)?;
        }
    } else {
        let output = &graph.output[0];
        let tensor = runner.get_storage(output.name())?;
        let tensor_bytes = tensor.read_bytes(&runner.device, &runner.queue);

        let elapsed = std::time::Instant::now().sub(time);
        log::info!("run done ({:?})", elapsed);

        let tensor_vec: &[f32] = bytemuck::cast_slice(&tensor_bytes);

        let out_shape = &descs[output.name()].shape;
        let filename = format!("activations/{}.npy", output.name());
        log::info!("saving to file {filename}");
        npy::save_to_file(&filename, tensor_vec, out_shape)
            .with_context(|| anyhow!("failed to save to file {filename}"))?;
    }
    log::info!("done!");

    Ok(())
}

