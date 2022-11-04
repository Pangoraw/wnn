use std::collections::HashMap;

use anyhow::{anyhow, Context};

use crate::{
    gpu::TensorDesc,
    onnx,
    shape::{self, Shape},
    tensor::DataType,
};

mod shape_inference;
mod type_inference;

pub(crate) type TensorDescs<'a> = HashMap<&'a str, TensorDesc>;

pub(crate) fn analyze_graph<'a>(
    graph: &'a onnx::GraphProto,
    dim_mappings: &HashMap<&str, shape::Dimension>,
) -> Result<TensorDescs<'a>, anyhow::Error> {
    let mut shape_inferer = shape_inference::ShapeInferer::new(&graph);
    let mut dtype_inferer = type_inference::TypeInferer::new();

    let mut descs = TensorDescs::new();

    for init in &graph.initializer {
        let shape = Shape::from(&init.dims);
        let dtype = DataType::from_int(init.data_type())?;
        dtype_inferer.init(init.name(), dtype.clone());
        shape_inferer.init(init.name(), shape.clone());
        descs.insert(init.name(), TensorDesc::new(shape, dtype));
    }

    println!("==== INPUT");
    for input in &graph.input {
        let dtype = DataType::from_int(input.type_.tensor_type().elem_type())?;
        let shape = {
            let tensor_shape =
                input.type_.tensor_type().shape.as_ref().ok_or_else(|| {
                    anyhow!("failed to get tensor shape for input {}", input.name())
                })?;
            Shape::from_tensor_shape_with_maps(tensor_shape, &dim_mappings)
        };
        println!("{}{}::{}", input.name(), &shape, &dtype);
        dtype_inferer.init(input.name(), dtype.clone());
        shape_inferer.init(input.name(), shape.clone());
        descs.insert(input.name(), TensorDesc::new(shape, dtype));
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
        let out_shapes = shape_inferer.infer_node(node).with_context(|| {
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
        for (out, (out_type, out_shape)) in node
            .output
            .iter()
            .zip(std::iter::zip(out_types, out_shapes))
        {
            intermediaries.push(out);
            shape_inferer.init(out, out_shape.clone());
            dtype_inferer.init(out, out_type.clone());
            descs.insert(out, TensorDesc::new(out_shape, out_type));
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

    Ok(descs)
}
