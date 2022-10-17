use std::collections::HashMap;

use anyhow::anyhow;

use crate::onnx;
use crate::tensor::DataType;
use crate::utils::get_attr_int;

pub(crate) struct TypeInferer<'a> {
    types: HashMap<&'a str, DataType>,
}

impl<'a> TypeInferer<'a> {
    pub(crate) fn new() -> Self {
        Self {
            types: HashMap::new(),
        }
    }

    pub(crate) fn infer_node(
        &mut self,
        node: &'a onnx::NodeProto,
    ) -> anyhow::Result<Vec<DataType>> {
        let out = match node.op_type() {
            "ConstantOfShape" => {
                let dtype = node
                    .attribute
                    .iter()
                    .find_map(|attr| match attr.name() {
                        "value" => Some(DataType::from_int(attr.t.data_type()).unwrap()),
                        _ => None,
                    })
                    .ok_or_else(|| anyhow!("could not infer type (no attr found)"))?;
                vec![dtype]
            }
            "Constant" => {
                let dtype = node
                    .attribute
                    .iter()
                    .find_map(|attr| match attr.name() {
                        "value_ints" => Some(DataType::F64),
                        "value_floats" => Some(DataType::F32),
                        "value" => Some(DataType::from_int(attr.t.data_type()).unwrap()),
                        "value_float" => Some(DataType::F32),
                        "value_int" => Some(DataType::I64),
                        _ => None,
                    })
                    .ok_or_else(|| anyhow!("could not infer type (no attr found)"))?;
                vec![dtype]
            }
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

    pub(crate) fn init(&mut self, name: &'a str, dtype: DataType) {
        self.types.insert(name, dtype);
    }

    pub(crate) fn get_type(&self, name: &str) -> &DataType {
        &self.types[name]
    }
}
