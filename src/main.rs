use std::collections::HashMap;

use anyhow::{anyhow, Context};
use protobuf::Message;
use structopt::StructOpt;

use crate::{
    compiler::is_reshape_op,
    gpu::{Op, TensorDesc},
    shape::Shape,
    tensor::DataType,
};

mod compiler;
mod gpu;
mod npy;
mod onnx;
mod shape;
mod shape_inference;
mod tensor;
mod type_inference;
mod utils;

#[derive(StructOpt, Debug)]
struct Args {
    #[structopt(default_value = "./sd-v1-5-onnx/vae_decoder_sim.onnx")]
    input_model: std::path::PathBuf,
    dump_folder: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
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

    let dim_mappings = {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "N",
            crate::shape::Dimension::Concrete(1),
            //  crate::shape::Dimension::Symbolic(String::from("batch")),
        );
        map.insert("channels", crate::shape::Dimension::Concrete(4));
        map.insert("height", crate::shape::Dimension::Concrete(64));
        map.insert("width", crate::shape::Dimension::Concrete(64));
        map
    };

    let graph = model.graph;
    let mut shape_inferer = shape_inference::ShapeInferer::new(&graph);
    let mut dtype_inferer = type_inference::TypeInferer::new();

    for init in &graph.initializer {
        let shape = Shape::from(&init.dims);
        let dtype = DataType::from_int(init.data_type())?;
        dtype_inferer.init(init.name(), dtype);
        shape_inferer.init(init.name(), shape.clone());
    }

    println!("==== INPUT");
    for input in &graph.input {
        let dtype = tensor::DataType::from_int(input.type_.tensor_type().elem_type())?;
        let shape = {
            let tensor_shape =
                input.type_.tensor_type().shape.as_ref().ok_or_else(|| {
                    anyhow!("failed to get tensor shape for input {}", input.name())
                })?;
            Shape::from_tensor_shape_with_maps(tensor_shape, &dim_mappings)
        };
        dtype_inferer.init(input.name(), dtype);
        shape_inferer.init(input.name(), shape);
    }

    let mut intermediaries = Vec::new();
    for node in &graph.node {
        let out_types = dtype_inferer.infer_node(node).with_context(|| {
            format!(
                "processing shapes for node {}[{}]({})",
                node.name(),
                node.op_type(),
                node.input
                    .iter()
                    .map(|input| shape_inferer.get_shape(input).to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            )
        })?;
        let out_shapes = shape_inferer.infer_node(node).with_context(|| {
            format!(
                "processing shapes for node {}[{}]({})",
                node.name(),
                node.op_type(),
                node.input
                    .iter()
                    .map(|input| shape_inferer.get_shape(input).to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            )
        })?;
        for (out, (out_type, out_shape)) in node
            .output
            .iter()
            .zip(std::iter::zip(out_types, out_shapes))
        {
            intermediaries.push(out);
            shape_inferer.init(out, out_shape);
            dtype_inferer.init(out, out_type);
        }

        if &std::env::var_os("DUMP_INFERENCE")
            .map(|s| s.to_str().unwrap().to_owned())
            .unwrap_or_else(|| String::from("1"))
            == "1"
        {
            for (i, out) in node.output.iter().enumerate() {
                print!(
                    "{}{}::{}",
                    out,
                    shape_inferer.get_shape(out),
                    dtype_inferer.get_type(out)
                );
                if i < node.output.len() - 1 {
                    print!(", ");
                }
            }
            print!("\t= {}[{}](", node.name(), node.op_type());
            for (i, input) in node.input.iter().enumerate() {
                if input.is_empty() {
                    print!("None");
                } else {
                    print!("{}{}", input, shape_inferer.get_shape(input));
                }
                if i < node.input.len() - 1 {
                    print!(", ");
                }
            }
            println!(")");
        }
    }

    println!("=== OUTPUT");
    let mut s = 0;
    for output in &graph.output {
        let computed_shape = shape_inferer.get_shape(output.name());
        let computed_type = dtype_inferer.get_type(output.name());
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
            let computed_shape = shape_inferer.get_shape(val.name());
            let info_shape = Shape::from_tensor_shape(shape);
            let computed_type = dtype_inferer.get_type(val.name());
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
    println!();
    println!("valided {}/{}", s, graph.node.len());

    if s != graph.node.len() {
        println!(
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

    // TODO: Move things out of main();

    let mut descs = HashMap::new(); // TODO: move this inside the runner/type inference phase
    let mut runner = pollster::block_on(gpu::Runner::new())?;

    for init in &graph.initializer {
        let dtype = dtype_inferer.get_type(init.name());
        let shape = shape_inferer.get_shape(init.name());
        let desc = TensorDesc::new(shape.clone(), dtype.clone());
        runner
            .add_init(init, desc.clone())
            .with_context(|| anyhow!("failed to create buffer for node {}", init.name()))?;
        descs.insert(init.name(), desc);
    }

    for input in &graph.input {
        let dtype = dtype_inferer.get_type(input.name());
        let shape = shape_inferer.get_shape(input.name());
        let desc = TensorDesc::new(shape.clone(), dtype.clone());
        let init = "arange";
        let numel = shape.numel().unwrap();
        let floats: Vec<f32> = if init == "ones" {
            std::iter::repeat([1.0]).flatten().take(numel).collect()
        } else {
            (0..numel).map(|i| i as f32).take(numel).collect()
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
        descs.insert(input.name(), desc);
    }

    for output in &graph.output {
        let dtype = dtype_inferer.get_type(output.name());
        let shape = shape_inferer.get_shape(output.name());
        let desc = TensorDesc::new(shape.clone(), dtype.clone());
        runner.add_node(output.name(), desc.clone(), true)?;
        descs.insert(output.name(), desc);
    }

    for inter in &intermediaries {
        let dtype = dtype_inferer.get_type(inter);
        let shape = shape_inferer.get_shape(inter);
        let desc = TensorDesc::new(shape.clone(), dtype.clone());
        let is_output = graph
            .output
            .iter()
            .any(|node| node.name() == inter.as_str());

        if !is_output {
            descs.insert(inter.as_str(), desc);
        }
    }
    runner
        .allocate_tensors(&graph.node, &descs, dump_folder.is_some())
        .with_context(|| anyhow!("when allocating nodes"))?;

    println!(
        "total_alloc_size = {}",
        human_bytes::human_bytes(runner.total_allocated_size() as f64)
    );
    // return Ok(());

    let ops = graph
        .node
        .iter()
        .filter(|node| !is_reshape_op(node.op_type()))
        .map(|node| {
            Op::new(
                &runner.device,
                node.input
                    .iter()
                    .filter_map(|input| {
                        if input.is_empty() {
                            None
                        } else {
                            Some(runner.get_storage(input))
                        }
                    })
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

        for inter in &intermediaries {
            let tensor = runner.get_storage(inter)?;

            let tensor_bytes = tensor.read_bytes(&runner.device, &runner.queue);
            let tensor_vec: &[f32] = bytemuck::cast_slice(&tensor_bytes);

            let inter_shape = &descs[inter.as_str()].shape;
            npy::save_to_file(&format!("activations/{inter}.npy"), tensor_vec, inter_shape)?;
        }
    } else {
        let output = &graph.output[0];
        let tensor = runner.get_storage(output.name())?;
        let tensor_bytes = tensor.read_bytes(&runner.device, &runner.queue);
        let tensor_vec: &[f32] = bytemuck::cast_slice(&tensor_bytes);

        let out_shape = &descs[output.name()].shape;
        npy::save_to_file(
            &format!("activations/{}.npy", output.name()),
            tensor_vec,
            out_shape,
        )
        .with_context(|| anyhow!("failed to save to file output.npy"))?;

        for f in tensor_vec.iter().take(20) {
            print!("{:1.4} ", f);
        }
        println!();
    }

    Ok(())
}
