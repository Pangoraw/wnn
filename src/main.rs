use anyhow::{anyhow, Context};
use protobuf::Message;

use crate::shape::Shape;

mod onnx;
mod shape;
mod shape_inference;
mod tensor;

fn main() -> anyhow::Result<()> {
    let mut onnx_file = std::fs::OpenOptions::new()
        .read(true)
        .open("/home/pberg/Downloads/model(1).onnx")?;
        // .open("./model.onnx")?;
        // .open("/home/pberg/irisa/diffusers/decoder_v1_4_pytorch_1_1.onnx")?;
    // .open("/home/pberg/Projects/ONNX.jl/model.onnx")?;
    // .open("/home/pberg/Projects/ONNX.jl/model_sim.onnx")?;
    let model = onnx::ModelProto::parse_from_reader(&mut onnx_file)?;

    let graph = model.graph;
    let mut inferer = shape_inference::ShapeInferer::new(&graph);

    for init in &graph.initializer {
        let shape = Shape::from(&init.dims);
        inferer.init(init.name(), shape);
    }

    for input in &graph.input {
        let shape =
            Shape::from_tensor_shape(
                input.type_.tensor_type().shape.as_ref().ok_or_else(|| {
                    anyhow!("failed to get tensor shape for input {}", input.name())
                })?,
            );
        inferer.init(input.name(), shape);
    }

    for node in &graph.node {
        let out_shapes = inferer.infer_node(node).with_context(|| {
            format!(
                "processing node {}[{}]({})",
                node.name(),
                node.op_type(),
                node.input
                    .iter()
                    .map(|input| inferer.get_shape(input).to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            )
        })?;
        for (out, out_shape) in std::iter::zip(&node.output, out_shapes) {
            inferer.init(out, out_shape);
        }

        for (i, out) in node.output.iter().enumerate() {
            print!("{}{}", out, inferer.get_shape(out));
            if i < node.output.len() - 1 {
                print!(", ");
            }
        }
        print!("\t= {}[{}](", node.name(), node.op_type());
        for (i, input) in node.input.iter().enumerate() {
            if input.is_empty() {
                print!("None");
            } else {
                print!("{}{}", input, inferer.get_shape(input));
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
        let computed_shape = inferer.get_shape(output.name());
        println!("{}{}", output.name(), computed_shape);
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
            let computed_shape = inferer.get_shape(val.name());
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

    Ok(())
}
