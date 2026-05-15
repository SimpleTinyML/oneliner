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

impl TensorSource for TensorRef {
    /// Reads an already materialized tensor descriptor.
    ///
    /// Input: pointer to a `TensorRef` descriptor slot.
    /// Output: the descriptor stored in that slot.
    ///
    /// Safety: `ptr` must point to an initialized `TensorRef`.
    unsafe fn tensor_ref_from_raw(ptr: *const Self) -> TensorRef {
        unsafe { core::ptr::read(ptr) }
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
/// Input: raw pointer to either a static byte array or a `TensorRef` descriptor.
/// Output: `TensorRef` consumed by `TensorRange`.
///
/// Safety: `ptr` must be valid for the concrete `T`.
pub unsafe fn tensor_ref_from_raw<T: TensorSource>(ptr: *const T) -> TensorRef {
    unsafe { T::tensor_ref_from_raw(ptr) }
}

/// Binds user input bytes to a generated input descriptor.
///
/// Input: descriptor slot, expected byte length, and user input bytes.
/// Output: `Ok(())` after the slot points to `input`, or size mismatch error.
///
/// Safety: `slot` must point to a valid mutable `TensorRef` descriptor.
pub unsafe fn bind_static_input(
    slot: *mut TensorRef,
    expected_len: usize,
    input: &[u8],
) -> Result<()> {
    if input.len() != expected_len {
        return Err(Error::InputSizeMismatch {
            provided: input.len(),
            expected: expected_len,
        });
    }
    unsafe {
        core::ptr::write(
            slot,
            TensorRef {
                ptr: input.as_ptr() as *mut u8,
                len: input.len(),
            },
        );
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
pub fn concurrent(commands: impl FnOnce()) {
    commands();
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
/// Input: target tensor range and scalar fill value.
/// Output: no value; writes into the target memory.
pub fn fill(target: TensorRange, value: impl FillValue) {
    let len = clipped_len(target);
    unsafe {
        core::ptr::write_bytes(target.tensor.ptr.add(target.offset), value.to_u8(), len);
    }
}

/// Clips a tensor range length so generated commands cannot run past the tensor.
///
/// Input: tensor range with offset and optional length.
/// Output: byte length available within the referenced tensor.
pub(crate) fn clipped_len(range: TensorRange) -> usize {
    let available = range.tensor.len.saturating_sub(range.offset);
    if range.length == 0 {
        available
    } else {
        range.length.min(available)
    }
}
