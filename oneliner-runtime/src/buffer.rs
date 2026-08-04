use super::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Ro,
    Wo,
    Rw,
    Unknown,
}

#[derive(Clone, Copy)]
pub struct Buffer {
    pub ptr: *const u8,
    pub len: usize,
}

impl Buffer {
    pub fn new(ptr: *const u8, len: usize) -> Self {
        Self::try_new(ptr, len).expect("invalid buffer reference")
    }

    fn try_new(ptr: *const u8, len: usize) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::NullPointer);
        }
        Ok(Self { ptr, len })
    }
}

#[derive(Clone, Copy)]
pub struct BufferMut {
    pub ptr: *mut u8,
    pub len: usize,
}

impl BufferMut {
    pub fn new(ptr: *mut u8, len: usize) -> Self {
        Self::try_new(ptr, len).expect("invalid buffer reference")
    }

    fn try_new(ptr: *mut u8, len: usize) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::NullPointer);
        }
        Ok(Self { ptr, len })
    }
}

/// Converts a generated storage item into the `BufferRef` used by dispatch.
///
/// Input: a raw pointer to the generated storage item.
/// Output: a `BufferRef` pointing at the buffer bytes.
pub trait BufferSource {
    /// Builds a `BufferRef` from a raw pointer to the implementing type.
    ///
    /// Input: pointer to converter-generated storage.
    /// Output: buffer pointer and byte length.
    ///
    /// Safety: `ptr` must point to a valid value of `Self`.
    unsafe fn to_buffer_ref(&self) -> Buffer;
    unsafe fn to_buffer_mut(&mut self) -> BufferMut;
}

impl<const N: usize> BufferSource for [u8; N] {
    /// Treats a static byte array as a buffer buffer.
    ///
    /// Input: pointer to `[u8; N]`.
    /// Output: `BufferRef` with the array pointer and fixed length `N`.
    ///
    /// Safety: `ptr` must point to a valid byte array.
    unsafe fn to_buffer_ref(&self) -> Buffer {
        Buffer {
            ptr: self as *const u8,
            len: N,
        }
    }
    unsafe fn to_buffer_mut(&mut self) -> BufferMut {
        BufferMut {
            ptr: self as *mut u8,
            len: N,
        }
    }
}

#[derive(Clone, Copy)]
pub struct BufferRange<T> {
    pub buffer: T,
    pub access: Access,
    pub offset: usize,
    pub length: usize,
}

#[derive(Clone, Copy)]
pub enum AnyBuffer {
    Ref(Buffer),
    Mut(BufferMut),
}

impl AnyBuffer {
    pub fn len(&self) -> usize {
        match self {
            AnyBuffer::Ref(t) => t.len,
            AnyBuffer::Mut(t) => t.len,
        }
    }
}

impl From<Buffer> for AnyBuffer {
    fn from(t: Buffer) -> Self {
        AnyBuffer::Ref(t)
    }
}

impl From<BufferMut> for AnyBuffer {
    fn from(t: BufferMut) -> Self {
        AnyBuffer::Mut(t)
    }
}

pub type AnyBufferRange = BufferRange<AnyBuffer>;

impl AnyBufferRange {
    pub fn new(buffer: AnyBuffer, access: Access, offset: usize, length: usize) -> Self {
        Self::try_new(buffer, access, offset, length).expect("invalid buffer range")
    }

    pub fn try_new(
        buffer: AnyBuffer,
        access: Access,
        offset: usize,
        length: usize,
    ) -> Result<Self, Error> {
        let buffer_len = buffer.len();

        let end = offset
            .checked_add(length)
            .ok_or(Error::BufferRangeOutOfBounds {
                offset,
                length,
                capacity: buffer_len,
            })?;

        if end > buffer_len {
            return Err(Error::BufferRangeOutOfBounds {
                offset,
                length,
                capacity: buffer_len,
            });
        }

        let access_valid = matches!(
            (buffer, access),
            (AnyBuffer::Ref(_), Access::Ro)
                | (AnyBuffer::Mut(_), Access::Ro)
                | (AnyBuffer::Mut(_), Access::Wo)
                | (AnyBuffer::Mut(_), Access::Rw)
        );

        if !access_valid {
            return Err(Error::InvalidAccess {
                access,
                required: match buffer {
                    AnyBuffer::Ref(_) => Access::Ro,
                    AnyBuffer::Mut(_) => Access::Rw,
                },
            });
        }

        Ok(Self {
            buffer: buffer,
            access,
            offset,
            length,
        })
    }
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
/// Output: low byte used to fill a buffer range.
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

/// Fills a generated buffer range with one byte value.
///
/// # Safety
///
/// The buffer pointer must be valid and writable for the declared range.
pub unsafe fn fill(target: AnyBufferRange, value: impl FillValue + Copy) -> Result<(), Error> {
    match target.buffer {
        AnyBuffer::Ref(_) => {
            return Err(Error::InvalidAccess {
                access: target.access,
                required: Access::Rw,
            });
        }
        AnyBuffer::Mut(buffer) => unsafe {
            core::ptr::write_bytes(buffer.ptr.add(target.offset), value.to_u8(), target.length);
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mutable_buffer(capacity: usize) -> AnyBuffer {
        BufferMut {
            ptr: core::ptr::dangling_mut(),
            len: capacity,
        }
        .into()
    }

    #[test]
    fn validates_buffer_ranges() {
        let range = AnyBufferRange::try_new(mutable_buffer(16), Access::Rw, 4, 8).unwrap();
        assert_eq!(range.offset, 4);
        assert_eq!(range.length, 8);

        assert!(AnyBufferRange::try_new(mutable_buffer(16), Access::Rw, 4, 0).is_ok());
        assert!(matches!(
            AnyBufferRange::try_new(mutable_buffer(16), Access::Rw, 17, 0),
            Err(Error::BufferRangeOutOfBounds { .. })
        ));
        assert!(matches!(
            AnyBufferRange::try_new(mutable_buffer(16), Access::Rw, 8, 9),
            Err(Error::BufferRangeOutOfBounds { .. })
        ));
    }
}
