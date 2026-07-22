use super::{Aligned, AlignedType};
use super::{Error, Prediction};

#[cfg(feature = "ariel-os")]
use ariel_os::log::{debug};

#[cfg(not(feature = "ariel-os"))]
use log::{debug};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Ro,
    Wo,
    Rw,
    Unknown,
}

#[derive(Clone, Copy)]
pub struct TensorRef {
    pub ptr: *const u8,
    pub len: usize,
}

impl TensorRef {
    pub fn new(ptr: *const u8, len: usize) -> Self {
        Self::try_new(ptr, len).expect("invalid tensor reference")
    }

    fn try_new(ptr: *const u8, len: usize) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::NullPointer);
        }
        Ok(Self { ptr, len })
    }
}

#[derive(Clone, Copy)]
pub struct TensorMut {
    pub ptr: *mut u8,
    pub len: usize,
}

impl TensorMut {
    pub fn new(ptr: *mut u8, len: usize) -> Self {
        Self::try_new(ptr, len).expect("invalid tensor reference")
    }

    fn try_new(ptr: *mut u8, len: usize) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::NullPointer);
        }
        Ok(Self { ptr, len })
    }
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
    unsafe fn to_tensor_ref(&self) -> TensorRef;
    unsafe fn to_tensor_mut(&mut self) -> TensorMut;
}

impl<const N: usize> TensorSource for [u8; N] {
    /// Treats a static byte array as a tensor buffer.
    ///
    /// Input: pointer to `[u8; N]`.
    /// Output: `TensorRef` with the array pointer and fixed length `N`.
    ///
    /// Safety: `ptr` must point to a valid byte array.
    unsafe fn to_tensor_ref(&self) -> TensorRef {
        TensorRef {
            ptr: self as *const u8,
            len: N,
        }
    }
    unsafe fn to_tensor_mut(&mut self) -> TensorMut {
        TensorMut {
            ptr: self as *mut u8,
            len: N,
        }
    }
}


#[derive(Clone, Copy)]
pub struct TensorRange<T> {
    pub tensor: T,
    pub access: Access,
    pub offset: usize,
    pub length: usize,

}

#[derive(Clone, Copy)]
pub enum AnyTensor {
    Ref(TensorRef),
    Mut(TensorMut),
}

impl AnyTensor {
    pub fn len(&self) -> usize {
        match self {
            AnyTensor::Ref(t) => t.len,
            AnyTensor::Mut(t) => t.len,
        }
    }
}

impl From<TensorRef> for AnyTensor {
    fn from(t: TensorRef) -> Self {
        AnyTensor::Ref(t)
    }
}

impl From<TensorMut> for AnyTensor {
    fn from(t: TensorMut) -> Self {
        AnyTensor::Mut(t)
    }
}


pub type AnyTensorRange = TensorRange<AnyTensor>;

impl AnyTensorRange {
    pub fn new(tensor: AnyTensor, access: Access, offset: usize, length: usize) -> Self {
        Self::try_new(tensor, access, offset, length)
            .expect("invalid tensor range")
    }

    pub fn try_new(
        tensor: AnyTensor,
        access: Access,
        offset: usize,
        length: usize,
    ) -> Result<Self, Error> {
        let tensor_len = tensor.len();

        let end = offset
            .checked_add(length)
            .ok_or(Error::TensorRangeOutOfBounds { offset, length, capacity: tensor_len })?;

        if end > tensor_len {
            return Err(Error::TensorRangeOutOfBounds { offset, length, capacity: tensor_len });
        }

        let access_valid = matches!(
            (tensor, access),
            (AnyTensor::Ref(_), Access::Ro)
                | (AnyTensor::Mut(_), Access::Ro)
                | (AnyTensor::Mut(_), Access::Wo)
                | (AnyTensor::Mut(_), Access::Rw)
        );

        if !access_valid {
            return Err(Error::InvalidAccess { access, required: match tensor {
                AnyTensor::Ref(_) => Access::Ro,
                AnyTensor::Mut(_) => Access::Rw,
            }});
        }

        Ok(Self {
            tensor,
            access,
            offset,
            length,
        })
    }

}

/// Copies user input into caller-owned aligned model storage.
///
/// Input: exclusively borrowed input storage and user input bytes.
/// Output: success when the input size exactly matches the generated storage.
pub fn write_input<const N: usize>(
    slot: &mut Aligned<AlignedType, [u8; N]>,
    input: &[u8],
) -> Result<(), Error> {
    if input.len() != N {
        return Err(Error::InputSizeMismatch {
            provided: input.len(),
            expected: N,
        });
    }
    slot.copy_from_slice(input);

    for i in 0..N {
        if slot[i] != input[i] {
            debug!(
                "write_input: mismatch at index {}: slot={} input={}",
                i, slot[i], input[i]
            );
        }
    }


    Ok(())
}

/// Borrows caller-owned aligned output storage as a `Prediction`.
///
/// Input: shared borrow of the generated output storage.
/// Output: prediction whose lifetime is tied to the storage borrow.
pub fn read_output<const N: usize>(src: &Aligned<AlignedType, [u8; N]>) -> Prediction<'_> {
    debug!("read_output: output={:?}", &src[..]);
    Prediction::from_slice(&src[..])
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
pub unsafe fn fill(target: AnyTensorRange, value: impl FillValue + Copy) -> Result<(), Error> {
    match target.tensor {
        AnyTensor::Ref(_) => {
            return Err(Error::InvalidAccess {
                access: target.access,
                required: Access::Rw,
            });
        }
        AnyTensor::Mut(tensor) => {
            unsafe {
                core::ptr::write_bytes(tensor.ptr.add(target.offset), value.to_u8(), target.length);
            }
        }
    }
    
    Ok(())
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
