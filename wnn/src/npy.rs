use anyhow::{bail, Result};
use std::io::{Read, Write};

use crate::shape::Shape;
use crate::tensor::{DataType, TensorDesc};

const MAGIC_STRING: &[u8; 6] = b"\x93NUMPY";
const SUPPORTED_VERSION: [u8; 2] = [
    0x01, // Major Version
    0x00, // Minor Version
];

/// Saves tensor data to the numpy file format
/// see https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
/// for more details.
pub fn save_to_file(filename: &str, data: &[u8], desc: &TensorDesc) -> Result<usize> {
    let shape = &desc.shape;

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(filename)?;

    if !shape.is_concrete() {
        bail!("cannot save not concrete shape {}", shape);
    }

    if !matches!(desc.dtype, DataType::F32) {
        bail!("only type f32 is currently supported");
    }

    let header = format!(
        "{{'descr': 'float32','fortran_order': False,'shape': {}}}\n",
        shape
    );
    let header = header.bytes().collect::<Vec<u8>>();
    let header_len: u16 = header.len() as u16;

    let mut n = MAGIC_STRING.len() + 2 + std::mem::size_of_val(&header_len) + header_len as usize;
    let padding = if n % 64 == 0 { 0 } else { 64 - n % 64 };
    let header_len = header_len + padding as u16;
    n += padding;

    assert!(n % 64 == 0);

    file.write_all(MAGIC_STRING)?;
    file.write_all(&SUPPORTED_VERSION)?;
    file.write_all(&header_len.to_le_bytes())?; // HEADER_LEN
    file.write_all(&header)?;

    file.write_all(&std::iter::repeat(b' ').take(padding).collect::<Vec<u8>>())?;
    n += data.len();

    file.write_all(data)?;

    Ok(n)
}

// Reading, needs a small python-dict parser

use nom::branch::alt;
use nom::bytes::complete::{tag, take, take_until};
use nom::character::complete::{char, i64 as nom_i64, space0};
use nom::combinator::{map, opt};
use nom::multi::separated_list0;
use nom::number::complete::{le_u16, u8 as nom_u8};
use nom::sequence::delimited;
use nom::IResult;

fn parse_header_len(data: &[u8]) -> IResult<&[u8], u16> {
    let (data, _) = tag(MAGIC_STRING)(data)?;
    let (data, major) = nom_u8(data)?;
    assert!(major == 1);

    let (data, minor) = nom_u8(data)?;
    assert!(minor == 0);

    le_u16(data)
}

fn extract_header(data: &[u8]) -> IResult<&[u8], &str> {
    let (data, header_len) = parse_header_len(data)?;
    let (data, header) = take(header_len)(data)?;
    let header = std::str::from_utf8(header).unwrap();

    Ok((data, header))
}

// We only support a subset of the things
#[derive(Debug)]
enum PythonVal {
    Bool(bool),
    Str(String),
    IntTuple(Vec<i64>),
}

fn parse_int_tuple(val: &str) -> IResult<&str, PythonVal> {
    let (rest, values) = delimited(
        char('('),
        separated_list0(char(','), delimited(space0, nom_i64, space0)),
        char(')'),
    )(val)?;
    Ok((rest, PythonVal::IntTuple(values)))
}

fn parse_bool(val: &str) -> IResult<&str, PythonVal> {
    let (rest, val) = alt((map(tag("True"), |_| true), map(tag("False"), |_| false)))(val)?;
    Ok((rest, PythonVal::Bool(val)))
}

fn parse_str(val: &str) -> IResult<&str, &str> {
    alt((
        delimited(char('\''), take_until("'"), char('\'')),
        delimited(char('"'), take_until("\""), char('"')),
    ))(val)
}

fn parse_string(val: &str) -> IResult<&str, PythonVal> {
    let (rest, s) = parse_str(val)?;
    Ok((rest, PythonVal::Str(s.to_owned())))
}

fn parse_val(val: &str) -> IResult<&str, PythonVal> {
    alt((parse_int_tuple, parse_string, parse_bool))(val)
}

fn parse_key_value(header: &str) -> IResult<&str, (&str, PythonVal)> {
    let (rest, _) = space0(header)?;
    let (rest, key) = parse_str(rest)?;
    let (rest, _) = char(':')(rest)?;
    let (rest, _) = space0(rest)?;
    let (rest, value) = parse_val(rest)?;

    Ok((rest, (key, value)))
}

fn parse_key_values(header: &str) -> IResult<&str, TensorDesc> {
    let (rest, key_values) = separated_list0(char(','), parse_key_value)(header)?;

    let mut slice: Option<Shape> = None;
    for (k, v) in key_values {
        match (k, &v) {
            ("descr", PythonVal::Str(s)) if s == "float32" || s == "<f4" => {}
            ("fortran_order", PythonVal::Bool(false)) => {}
            ("shape", PythonVal::IntTuple(s)) => slice = Some(Shape::from(s)),
            _ => {
                panic!("{} {:?}", k, v)
            }
        }
    }

    let (rest, _) = opt(space0)(rest)?;
    let (rest, _) = opt(char(','))(rest)?;
    let (rest, _) = opt(space0)(rest)?;

    Ok((rest, TensorDesc::new(slice.unwrap(), DataType::F32)))
}

fn parse_header(header: &str) -> IResult<&str, TensorDesc> {
    delimited(char('{'), parse_key_values, char('}'))(header)
}

pub(crate) fn read_from_bytes(bytes: &[u8]) -> Result<(TensorDesc, &[u8])> {
    let Ok((slice, header)) = extract_header(bytes) else {
        bail!("failed to parse file header");
    };
    let Ok((_, desc)) = parse_header(header) else {
        bail!("failed to parse header");
    };

    if slice.len() != desc.size_of() {
        bail!(
            "invalid data slice of size {} for shape {} of type {}",
            slice.len(),
            desc.shape,
            desc.dtype
        );
    }

    Ok((desc, slice))
}

pub fn read_from_file(filename: &str) -> Result<(TensorDesc, Vec<u8>)> {
    let mut file = std::fs::OpenOptions::new().read(true).open(filename)?;
    let mut content = Vec::new();
    file.read_to_end(&mut content)?;

    let (desc, slice) = read_from_bytes(&content)?;

    Ok((desc, slice.to_vec()))
}
