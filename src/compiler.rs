// This file is very much inspired/copied from webonnx/wonnx

use std::collections::HashMap;

use anyhow::anyhow;
use lazy_static::lazy_static;

use crate::{gpu::TensorDesc, onnx};

lazy_static! {
    static ref SHADER_FILES: HashMap<&'static str, &'static str> = HashMap::from_iter([
        ("matmul", include_str!("../shaders/matmul.wgsl")),
        ("relu", include_str!("../shaders/relu.wgsl"))
    ]);
}

/// We should be able to convert to wgsl String
/// and get the workgroup dispatch informations.
pub(crate) struct ShaderInvocation {
    file_name: &'static str,
    context: tera::Context,
    dispatch: (u32, u32, u32),
}

impl ShaderInvocation {
    pub(crate) fn to_wgsl(&self) -> anyhow::Result<String> {
        let template = SHADER_FILES
            .get(&self.file_name)
            .ok_or_else(|| anyhow!("invalid template {}", self.file_name))?;
        Ok(tera::Tera::one_off(template, &self.context, false)?)
    }

    pub(crate) fn dispatch(&self) -> (u32, u32, u32) {
        self.dispatch
    }
}

fn base_context(
    node: &onnx::NodeProto,
    descs: &HashMap<&str, TensorDesc>,
) -> anyhow::Result<tera::Context> {
    let mut context = tera::Context::new();
    context.insert(
        "i_sizes",
        &node
            .input
            .iter()
            .map(|input| descs[input.as_str()].shape.as_ints())
            .collect::<anyhow::Result<Vec<Vec<i64>>>>()?,
    );
    context.insert(
        "o_sizes",
        &node
            .output
            .iter()
            .map(|input| descs[input.as_str()].shape.as_ints())
            .collect::<anyhow::Result<Vec<Vec<i64>>>>()?,
    );
    Ok(context)
}

pub(crate) fn compile_node(
    node: &onnx::NodeProto,
    descs: &HashMap<&str, TensorDesc>,
) -> anyhow::Result<ShaderInvocation> {
    let mut context = base_context(node, descs)?;
    let invoc = match node.op_type() {
        "MatMul" => {
            let workgroup_y = descs[node.input[0].as_str()].shape.concrete_size(0)?;
            let workgroup_x = descs[node.input[1].as_str()].shape.concrete_size(1)?;

            context.insert("workgroup_y", &workgroup_y);
            context.insert("workgroup_x", &workgroup_x);

            ShaderInvocation {
                file_name: "matmul",
                context,
                dispatch: (1, 1, 1),
            }
        }
        "Relu" => {
            let out_shape = &descs[node.output[0].as_str()].shape;

            ShaderInvocation {
                file_name: "relu",
                context,
                dispatch: (
                    out_shape
                        .numel()
                        .ok_or_else(|| anyhow!("could not compute arity"))?
                        as _,
                    1,
                    1,
                ),
            }
        }
        op => unimplemented!("{op}"),
    };

    Ok(invoc)
}
