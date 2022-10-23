use anyhow::{anyhow, bail, Result};
use std::io::{Read, Write};

use crate::shape::Shape;

const MAGIC_STRING: [u8; 6] = [0x93, 'N' as u8, 'U' as u8, 'M' as u8, 'P' as u8, 'Y' as u8];
const SUPPORTED_VERSION: [u8; 2] = [
    0x01, // Major Version
    0x00, // Minor Version
];

/// Saves tensor data to the numpy file format
/// see https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
/// for more details.
pub(crate) fn save_to_file(filename: &str, data: &[f32], shape: &Shape) -> Result<usize> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(filename)?;

    if !shape.is_concrete() {
        bail!("cannot save not concrete shape {}", shape);
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

    file.write_all(&MAGIC_STRING)?;
    file.write_all(&SUPPORTED_VERSION)?;
    file.write_all(&header_len.to_le_bytes())?; // HEADER_LEN
    file.write_all(&header)?;

    file.write_all(
        &std::iter::repeat(' ' as u8)
            .take(padding)
            .collect::<Vec<u8>>(),
    )?;
    let data_slice = bytemuck::cast_slice(data);
    n += data_slice.len();

    file.write_all(data_slice)?;

    Ok(n)
}

pub(crate) fn read_from_file(filename: &str) -> Result<(Shape, Vec<f32>)> {
    let mut file = std::fs::OpenOptions::new().read(true).open(filename)?;
    let mut content = Vec::new();
    file.read_to_end(&mut content)?;

    assert!(&content[0..6] == &MAGIC_STRING);
    assert!(&content[6..8] == &SUPPORTED_VERSION);

    let header_offset = 8 + std::mem::size_of::<u16>();
    let header_len_bytes: &[u8; 2] = &content[8..header_offset].try_into()?;
    let header_len = u16::from_le_bytes(*header_len_bytes);

    let header = std::str::from_utf8(&content[header_offset..header_offset + header_len as usize])?;

    let mut shape: Option<Shape> = None;
    match header
        .trim_end()
        .strip_prefix('{')
        .map(|header| header.strip_suffix('}'))
    {
        Some(Some(header)) => {
            for key_value in header.split(",") {
                let (key, value) = {
                    let key_value = key_value.split(':').collect::<Vec<&str>>();
                    println!("{:?}", key_value);
                    (key_value[0].trim(), key_value[1].trim())
                };

                match key {
                    "'fortran_order'" => assert!(value == "False"),
                    "'descr'" => assert!(value == "'float32'" || value == "'<f8'"),
                    "'shape'" => {
                        if let Some(Some(val)) = value
                            .strip_prefix('(')
                            .map(|header| header.strip_suffix(')'))
                        {
                            let ints = val
                                .split(',')
                                .map(|s| i64::from_str_radix(s.trim(), 10))
                                .collect::<std::result::Result<Vec<i64>, std::num::ParseIntError>>(
                            )?;
                            shape = Some(Shape::from(&ints));
                        }
                    }
                    _ => bail!("invalid key {key} in header"),
                }
            }
        }
        _ => bail!("invalid header '{header}'"),
    }

    let data = vec![1.];
    match shape {
        Some(s) => Ok((s, data)),
        _ => Err(anyhow!("could not find shape")),
    }
}
