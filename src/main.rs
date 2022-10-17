use anyhow::{anyhow, Context};
use protobuf::Message;

use crate::{
    gpu::{Op, TensorDesc},
    shape::Shape,
    tensor::DataType,
};

mod gpu;
mod onnx;
mod ops;
mod shape;
mod shape_inference;
mod tensor;
mod type_inference;
mod utils;

fn main() -> anyhow::Result<()> {
    let filename = "./simple_model.onnx";
    // let filename = "vae_decoder_sim.onnx";
    // let filename = "unet.onnx";
    // let filename = "./model.onnx";
    // let filename = "/home/pberg/irisa/diffusers/decoder_v1_4_pytorch_1_1.onnx";
    // let filename = "/home/pberg/Projects/ONNX.jl/model.onnx";
    // let filename = "/home/pberg/Projects/ONNX.jl/model_sim.onnx";

    let mut onnx_file = std::fs::OpenOptions::new()
        .read(true)
        .open(filename)
        .with_context(|| format!("while opening file {}", filename))?;
    let model = onnx::ModelProto::parse_from_reader(&mut onnx_file)?;

    let dim_mappings = {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "batch",
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
        shape_inferer.init(init.name(), shape);
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
        println!("input {} {}", input.name(), shape);
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
        println!(")")
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
            if computed_shape != &info_shape {
                println!(
                    "warning: {}: computed {}, stored {}",
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

    let mut total_alloc_size = 0;
    let max_alloc_size = 7_500_000_000; // TODO: get GPU capacity

    let mut runner = pollster::block_on(gpu::Runner::new())?;
    for init in &graph.initializer {
        let dtype = dtype_inferer.get_type(init.name());
        let shape = shape_inferer.get_shape(init.name());
        let desc = TensorDesc::new(shape.clone(), dtype.clone());
        total_alloc_size += desc.size_of();
        let pretty = human_bytes::human_bytes(desc.size_of() as u32);
        println!(
            "allocating {} {}",
            pretty,
            human_bytes::human_bytes(total_alloc_size as f64)
        );
        runner.add_init(init, desc)?;

        if total_alloc_size > max_alloc_size {
            anyhow::bail!(
                "stopping at node {} when allocating {}",
                init.name(),
                pretty
            );
        }
    }

    for inter in &intermediaries {
        let dtype = dtype_inferer.get_type(inter);
        let shape = shape_inferer.get_shape(inter);
        let desc = TensorDesc::new(shape.clone(), dtype.clone());
        total_alloc_size += desc.size_of();
        let pretty = human_bytes::human_bytes(desc.size_of() as u32);
        let is_output = graph
            .output
            .iter()
            .any(|node| node.name() == inter.as_str());

        println!(
            "allocating {} {} {}",
            pretty,
            human_bytes::human_bytes(total_alloc_size as f64),
            is_output
        );
        runner.add_node(inter, desc, is_output)?;

        if total_alloc_size > max_alloc_size {
            anyhow::bail!("stopping at node {} when allocating {}", inter, pretty);
        }
    }

    for input in &graph.input {
        let dtype = dtype_inferer.get_type(input.name());
        let shape = shape_inferer.get_shape(input.name());
        let desc = TensorDesc::new(shape.clone(), dtype.clone());
        total_alloc_size += desc.size_of();
        let pretty = human_bytes::human_bytes(desc.size_of() as u32);
        println!(
            "allocating {} {}",
            pretty,
            human_bytes::human_bytes(total_alloc_size as f64)
        );
        let floats = [1., 1.];
        let data = unsafe {
            let ptr = floats.as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr, 2 * 4)
        };

        runner.add_node_with_init(input.name(), desc, data)?;
        if total_alloc_size > max_alloc_size {
            anyhow::bail!(
                "stopping at node {} when allocating {}",
                input.name(),
                pretty
            );
        }
    }

    let node = &graph.node[0];
    let op = Op::new(
        &runner.device,
        "matmul",
        node.input
            .iter()
            .map(|input| runner.get_storage(input))
            .collect(),
        node.output
            .iter()
            .map(|output| runner.get_storage(output))
            .collect(),
    );

    let mut encoder = runner
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Command Encoder"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
        });

        op.run(&mut compute_pass)?;
    }
    runner.queue.submit(std::iter::once(encoder.finish()));

    println!("created op");

    let tensor = runner.get_storage(graph.output[0].name());
    let tensor_bytes = tensor.to_bytes(&runner.device);
    let tensor_vec = unsafe {
        let ptr = tensor_bytes.as_ptr() as *const f32;
        std::slice::from_raw_parts(ptr, tensor_bytes.len() / 4)
    };

    println!("{:?}", tensor_vec);

    Ok(())
}
