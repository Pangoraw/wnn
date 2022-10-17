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

// There is sometimes an issue where the data is not actually in the
// tensor.T_data field so we use raw_data instead.
pub(crate) fn int_slice_from_tensor(tensor: &onnx::TensorProto) -> &[i64] {
    if !tensor.int64_data.is_empty() {
        &tensor.int64_data
    } else {
        let raw_data = tensor.raw_data();
        let slice = unsafe {
            let ptr = raw_data.as_ptr() as *const i64;
            std::slice::from_raw_parts(ptr, raw_data.len() / std::mem::size_of::<i64>())
        };
        slice
    }
}

pub(crate) fn float_slice_from_tensor(tensor: &onnx::TensorProto) -> &[f32] {
    if !tensor.float_data.is_empty() {
        &tensor.float_data
    } else {
        let raw_data = tensor.raw_data();
        unsafe {
            let ptr = raw_data.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, raw_data.len() / std::mem::size_of::<f32>())
        }
    }
}
