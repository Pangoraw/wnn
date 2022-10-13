use std::collections::HashMap;

use onnx::AttributeProto;
use protobuf::Message;

use crate::{onnx::tensor_proto::DataType, shape::Shape};

mod onnx;
mod shape;
mod tensor;

fn infer_shapes(
    shapes: &mut std::collections::HashMap<String, Shape>,
    constants: &mut std::collections::HashMap<String, Shape>,
    node: &onnx::NodeProto,
) -> Vec<Shape> {
    match node.op_type() {
        "Gemm" => {
            let input_shapes = node
                .input
                .iter()
                .map(|input| shapes.get(input.as_str()).unwrap())
                .collect::<Vec<&Shape>>();
            assert!(input_shapes.len() == 3);

            let input = input_shapes[0];
            let weights = input_shapes[1];
            let bias = input_shapes[2];

            let mut out_dim = Shape::empty();
            if input.size(-1) == weights.size(-1)
                && bias.ndims() == 1
                && bias.size(0) == weights.size(-2)
            {
                for dim in 0isize..input.ndims() as isize - 1 {
                    out_dim.add_dim(dim, input.size(dim).clone());
                }
                out_dim.append_dim(weights.size(-2).clone());
            } else {
                unreachable!("invalid Gemm");
            }
            vec![out_dim]
        }
        op @ ("Mul" | "Add" | "Div" | "Sub") => {
            let input_shapes = node
                .input
                .iter()
                .map(|input| shapes.get(input.as_str()).unwrap())
                .collect::<Vec<&Shape>>();

            let a = input_shapes[0];
            let b = input_shapes[1];

            let out_shape = if a.is_scalar() {
                b.clone()
            } else if b.is_scalar() {
                a.clone()
            } else {
                println!("{a} * {b}");
                unimplemented!("{op}");
            };

            vec![out_shape]
        }
        "Conv" => {
            let input_shapes = node
                .input
                .iter()
                .map(|input| shapes.get(input.as_str()).unwrap())
                .collect::<Vec<&Shape>>();
            let x = input_shapes[0];
            let w = input_shapes[1];

            assert!(x.ndims() == 4 && w.ndims() == 4);

            let mut out_shape = Shape::empty();
            out_shape.append_dim(x.size(0).clone());
            out_shape.append_dim(w.size(0).clone());

            out_shape.append_dim(x.size(2).clone());
            out_shape.append_dim(x.size(3).clone());

            vec![out_shape]
        }
        "Reshape" => {
            let shape = match constants.get(&node.input[1]) {
                Some(s) => s,
                None => &shapes[&node.input[1]],
            };
            let input_shape = &shapes[&node.input[0]];
            let out_shape = shape.map_and_rest(&input_shape);
            vec![out_shape]
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
                            let raw_data = attr.t.raw_data();
                            let real_size = unsafe {
                                let ptr = raw_data.as_ptr() as *const i64;
                                std::slice::from_raw_parts(ptr, raw_data.len() / 8)
                            };
                            constants.insert(node.output[0].to_string(), Shape::from(real_size));
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

            vec![shape.unwrap()]
        }
        "Shape" => {
            let input_shape = &shapes[&node.input[0]];
            constants.insert(node.output[0].to_string(), input_shape.clone());
            vec![Shape::from(&[input_shape.ndims() as _])]
        }
        "InstanceNormalization" | "Cast" | "Relu" | "Sigmoid" | "Tanh" | "Identity" => {
            vec![shapes[node.input[0].as_str()].clone()]
        }
        other => unimplemented!("node type {}", other),
    }
}

fn main() {
    let mut onnx_file = std::fs::OpenOptions::new()
        .read(true)
        .open("/home/paul/Downloads/vae_decoder.onnx")
        .unwrap();
    let model = onnx::ModelProto::parse_from_reader(&mut onnx_file).unwrap();

    let graph = model.graph;
    let mut shapes: HashMap<String, Shape> = std::collections::HashMap::new();

    for init in &graph.initializer {
        let shape = Shape::from(&init.dims);
        shapes.insert(init.name().to_string(), shape);
    }

    for input in &graph.input {
        let shape = Shape::from_tensor_shape(input.type_.tensor_type().shape.as_ref().unwrap());
        shapes.insert(input.name().to_string(), shape);
    }
    let mut constants = HashMap::new();

    for node in &graph.node {
        let out_shapes = infer_shapes(&mut shapes, &mut constants, node);
        for (out, out_shape) in std::iter::zip(&node.output, out_shapes) {
            shapes.insert(out.to_string(), out_shape);
        }

        for out in &node.output {
            print!("{}{}, ", out, shapes[out]);
        }
        print!("\t= {}(", node.name.as_ref().unwrap(),);
        for input in &node.input {
            print!("{}{}, ", input, shapes[input]);
        }
        println!(")")
    }

    // dbg!(&shapes);

    // dbg!(graph);
}
