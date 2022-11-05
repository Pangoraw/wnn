use crate::onnx;

pub(crate) fn get_attr_int(node: &onnx::NodeProto, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find_map(|attr| if attr.name() == name { attr.i } else { None })
}

pub(crate) fn get_attr_ints<'a>(node: &'a onnx::NodeProto, name: &str) -> Option<&'a [i64]> {
    node.attribute.iter().find_map(|attr| {
        if attr.name() == name {
            Some(attr.ints.as_ref())
        } else {
            None
        }
    })
}

pub(crate) fn get_attr_float(node: &onnx::NodeProto, name: &str) -> Option<f32> {
    node.attribute
        .iter()
        .find_map(|attr| if attr.name() == name { attr.f } else { None })
}

pub(crate) fn get_attr_string<'a>(node: &'a onnx::NodeProto, name: &str) -> Option<&'a str> {
    node.attribute.iter().find_map(|attr| {
        if attr.name() == name {
            std::str::from_utf8(attr.s()).ok()
        } else {
            None
        }
    })
}

pub(crate) fn _external_data(tensor: &onnx::TensorProto) {
    if matches!(
        tensor.data_location(),
        onnx::tensor_proto::DataLocation::EXTERNAL
    ) {
        for key in &tensor.external_data {
            println!("{} => {}", key.key(), key.value());
        }
    }
}

// There is sometimes an issue where the data is not actually in the
// tensor.T_data field so we use raw_data instead.
pub(crate) fn int_slice_from_tensor(tensor: &onnx::TensorProto) -> &[i64] {
    if !tensor.int64_data.is_empty() {
        &tensor.int64_data
    } else {
        bytemuck::cast_slice(tensor.raw_data())
    }
}

pub(crate) fn float_slice_from_tensor(tensor: &onnx::TensorProto) -> &[f32] {
    if !tensor.float_data.is_empty() {
        &tensor.float_data
    } else {
        bytemuck::cast_slice(tensor.raw_data())
    }
}
