use std::collections::HashMap;

use anyhow::anyhow;

use crate::onnx;
use crate::tensor::DataType;
use crate::utils::get_attr_int;

pub(super) struct TypeInferer<'a> {
    types: HashMap<&'a str, DataType>,
}

impl<'a> TypeInferer<'a> {
    pub(super) fn new() -> Self {
        Self {
            types: HashMap::new(),
        }
    }

    pub(super) fn infer_node(
        &mut self,
        node: &'a onnx::NodeProto,
    ) -> anyhow::Result<Vec<DataType>> {
        let out = match node.op_type() {
            "ConstantOfShape" => {
                let dtype = node
                    .attribute
                    .iter()
                    .find_map(|attr| match attr.name() {
                        "value" => Some(DataType::from_int(attr.t.data_type())),
                        _ => None,
                    })
                    .unwrap_or_else(|| Ok(DataType::F32))?;
                vec![dtype]
            }
            "Constant" => {
                let dtype = node
                    .attribute
                    .iter()
                    .find_map(|attr| match attr.name() {
                        "value_ints" => Some(Ok(DataType::F64)),
                        "value_floats" => Some(Ok(DataType::F32)),
                        "value" => Some(DataType::from_int(attr.t.data_type())),
                        "value_float" => Some(Ok(DataType::F32)),
                        "value_int" => Some(Ok(DataType::I64)),
                        _ => None,
                    })
                    .unwrap_or_else(|| Err(anyhow!("could not infer type (no attr found)")))?;
                vec![dtype]
            }
            "Shape" => vec![DataType::I64],
            "Cast" => {
                let to = get_attr_int(node, "to")
                    .ok_or_else(|| anyhow!("failed to get to attribute"))?;

                vec![DataType::from_int(to as _)?]
            }
            _ => {
                let out = match self.types.get(node.input[0].as_str()) {
                    Some(dtype) => dtype.clone(),
                    None => anyhow::bail!("failed to find type for {}", node.input[0]),
                };
                vec![out]
            }
        };
        Ok(out)
    }

    pub(super) fn init(&mut self, name: &'a str, dtype: DataType) {
        self.types.insert(name, dtype);
    }

    pub(super) fn get_type(&self, name: &str) -> &DataType {
        &self.types[name]
    }
}
