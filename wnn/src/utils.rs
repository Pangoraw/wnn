use std::io::{Read, Seek};

use anyhow::{bail, Context};

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

pub(crate) fn get_attr_floats<'a>(node: &'a onnx::NodeProto, name: &str) -> Option<&'a [f32]> {
    node.attribute.iter().find_map(|attr| {
        if attr.name() == name {
            Some(attr.floats.as_ref())
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

/// Reads the tensor data into a vector from the external file it is stored in.
pub(crate) fn external_data(tensor: &onnx::TensorProto) -> anyhow::Result<Vec<u8>> {
    assert!(tensor.data_location() == onnx::tensor_proto::DataLocation::EXTERNAL);

    let mut location = "";
    let mut offset = 0;
    let mut bytes_length = 0;

    for keyval in &tensor.external_data {
        match keyval.key() {
            "location" => location = keyval.value(),
            "offset" => offset = keyval.value().parse::<u64>()?,
            "length" => bytes_length = keyval.value().parse::<usize>()?,
            other => bail!("invalid key '{other}'"),
        }
    }

    // TODO: Optimize by keeping the file descriptor open
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .open(format!("/home/paul/Projects/ONNX.jl/{location}"))
        .with_context(|| anyhow::anyhow!("when open file '{location}'"))?;
    file.seek(std::io::SeekFrom::Start(offset))?;

    let mut content = vec![0; bytes_length];
    file.read_exact(&mut content)?;

    Ok(content)
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
