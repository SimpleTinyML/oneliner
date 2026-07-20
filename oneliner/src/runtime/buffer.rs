use super::{Aligned, AlignedType};
use super::{Error, Prediction, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Ro,
    Wo,
    Rw,
    Unknown,
}

#[derive(Clone, Copy)]
pub struct TensorRef {
    pub ptr: *mut u8,
    pub len: usize,
}

/// Converts a generated storage item into the `TensorRef` used by dispatch.
///
/// Input: a raw pointer to the generated storage item.
/// Output: a `TensorRef` pointing at the tensor bytes.
pub trait TensorSource {
    /// Builds a `TensorRef` from a raw pointer to the implementing type.
    ///
    /// Input: pointer to converter-generated storage.
    /// Output: tensor pointer and byte length.
    ///
    /// Safety: `ptr` must point to a valid value of `Self`.
    unsafe fn tensor_ref_from_raw(ptr: *const Self) -> TensorRef;
}

impl<const N: usize> TensorSource for [u8; N] {
    /// Treats a static byte array as a tensor buffer.
    ///
    /// Input: pointer to `[u8; N]`.
    /// Output: `TensorRef` with the array pointer and fixed length `N`.
    ///
    /// Safety: `ptr` must point to a valid byte array.
    unsafe fn tensor_ref_from_raw(ptr: *const Self) -> TensorRef {
        TensorRef {
            ptr: ptr as *mut u8,
            len: N,
        }
    }
}

impl<const N: usize> TensorSource for Aligned<AlignedType, [u8; N]> {
    /// Treats an aligned static byte array as a tensor buffer.
    ///
    /// Input: pointer to `Aligned<AlignedType, [u8; N]>`.
    /// Output: `TensorRef` with the array pointer and fixed length `N`.
    ///
    /// Safety: `ptr` must point to a valid aligned byte array.
    unsafe fn tensor_ref_from_raw(ptr: *const Self) -> TensorRef {
        TensorRef {
            ptr: ptr as *mut u8,
            len: N,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TensorRange {
    pub tensor: TensorRef,
    pub access: Access,
    pub offset: usize,
    pub length: usize,
}

/// Converts generated tensor storage into a dispatch-ready `TensorRef`.
///
/// Input: raw pointer to converter-generated static byte-array storage.
/// Output: `TensorRef` consumed by `TensorRange`.
///
/// Safety: `ptr` must be valid for the concrete `T`.
pub unsafe fn tensor_ref_from_raw<T: TensorSource>(ptr: *const T) -> TensorRef {
    unsafe { T::tensor_ref_from_raw(ptr) }
}

/// Copies user input into generated aligned storage.
///
/// # Safety
///
/// `slot` must point to valid, exclusively writable input storage.
pub unsafe fn write_static_input<const N: usize>(
    slot: *mut Aligned<AlignedType, [u8; N]>,
    input: &[u8],
) -> Result<()> {
    if input.len() != N {
        return Err(Error::InputSizeMismatch {
            provided: input.len(),
            expected: N,
        });
    }
    unsafe {
        let destination = core::ptr::addr_of_mut!((*slot)) as *mut u8;
        core::ptr::copy_nonoverlapping(input.as_ptr(), destination, N);
    }
    Ok(())
}

/// Wraps a static output buffer as a `Prediction`.
///
/// Input: pointer and byte length for the output buffer.
/// Output: borrowed prediction view over that output memory.
///
/// Safety: `src..src + len` must be valid readable memory for the prediction lifetime.
pub unsafe fn read_static_output(src: *const u8, len: usize) -> Prediction<'static> {
    Prediction::from_slice(unsafe { core::slice::from_raw_parts(src, len) })
}

/// Runs a group of generated commands sequentially.
///
/// Input: closure containing generated command calls.
/// Output: no value; all effects are performed by the closure.
pub fn concurrent<T>(commands: impl FnOnce() -> T) -> T {
    commands()
}

/// Converts scalar fill values to the byte written by `fill`.
///
/// Input: scalar value.
/// Output: low byte used to fill a tensor range.
pub trait FillValue {
    /// Converts this scalar into a byte fill value.
    ///
    /// Input: `self`.
    /// Output: `u8` value passed to `write_bytes`.
    fn to_u8(self) -> u8;
}

macro_rules! impl_fill_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl FillValue for $ty {
                /// Converts this integer into the byte used by `fill`.
                ///
                /// Input: integer value.
                /// Output: low byte of the integer.
                fn to_u8(self) -> u8 {
                    self as u8
                }
            }
        )*
    };
}

impl_fill_value!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// Fills a generated tensor range with one byte value.
///
/// # Safety
///
/// The tensor pointer must be valid and writable for the declared range.
pub unsafe fn fill(target: TensorRange, value: impl FillValue) -> Result<()> {
    let len = checked_len(target)?;
    if len == 0 {
        return Ok(());
    }
    unsafe {
        core::ptr::write_bytes(target.tensor.ptr.add(target.offset), value.to_u8(), len);
    }
    Ok(())
}

pub(crate) fn checked_len(range: TensorRange) -> Result<usize> {
    if range.offset > range.tensor.len {
        return Err(Error::TensorRangeOutOfBounds {
            offset: range.offset,
            length: range.length,
            capacity: range.tensor.len,
        });
    }
    let available = range.tensor.len - range.offset;
    let length = if range.length == 0 {
        available
    } else {
        range.length
    };
    if length > available {
        return Err(Error::TensorRangeOutOfBounds {
            offset: range.offset,
            length,
            capacity: range.tensor.len,
        });
    }
    Ok(length)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(capacity: usize, offset: usize, length: usize) -> TensorRange {
        TensorRange {
            tensor: TensorRef {
                ptr: core::ptr::dangling_mut(),
                len: capacity,
            },
            access: Access::Rw,
            offset,
            length,
        }
    }

    #[test]
    fn validates_tensor_ranges() {
        assert_eq!(checked_len(range(16, 4, 8)), Ok(8));
        assert_eq!(checked_len(range(16, 4, 0)), Ok(12));
        assert!(matches!(
            checked_len(range(16, 17, 0)),
            Err(Error::TensorRangeOutOfBounds { .. })
        ));
        assert!(matches!(
            checked_len(range(16, 8, 9)),
            Err(Error::TensorRangeOutOfBounds { .. })
        ));
    }
}
