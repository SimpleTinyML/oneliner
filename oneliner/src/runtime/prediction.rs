#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
enum PredictionBytes<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prediction<'a> {
    #[cfg(feature = "alloc")]
    bytes: PredictionBytes<'a>,
    #[cfg(not(feature = "alloc"))]
    bytes: &'a [u8],
}

impl<'a> Prediction<'a> {
    /// Creates a borrowed prediction from output bytes.
    ///
    /// Input: byte slice containing model output.
    /// Output: `Prediction` borrowing the slice.
    pub const fn from_slice(bytes: &'a [u8]) -> Self {
        Self {
            #[cfg(feature = "alloc")]
            bytes: PredictionBytes::Borrowed(bytes),
            #[cfg(not(feature = "alloc"))]
            bytes,
        }
    }

    /// Creates an empty prediction.
    ///
    /// Input: none.
    /// Output: prediction whose byte view is empty.
    pub const fn empty() -> Self {
        Self::from_slice(&[])
    }

    /// Returns prediction bytes.
    ///
    /// Input: borrowed prediction.
    /// Output: borrowed byte slice containing the prediction payload.
    pub fn as_bytes(&self) -> &[u8] {
        #[cfg(feature = "alloc")]
        {
            match &self.bytes {
                PredictionBytes::Borrowed(bytes) => bytes,
                PredictionBytes::Owned(bytes) => bytes.as_slice(),
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.bytes
        }
    }
}

#[cfg(feature = "alloc")]
impl Prediction<'static> {
    /// Creates an owned prediction from a byte vector.
    ///
    /// Input: vector containing prediction bytes.
    /// Output: `Prediction` owning the vector.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes: PredictionBytes::Owned(bytes),
        }
    }

    /// Converts a prediction into owned bytes.
    ///
    /// Input: prediction that may borrow or own bytes.
    /// Output: owned `Vec<u8>` containing the prediction payload.
    pub fn into_bytes(self) -> Vec<u8> {
        match self.bytes {
            PredictionBytes::Borrowed(bytes) => bytes.to_vec(),
            PredictionBytes::Owned(bytes) => bytes,
        }
    }
}

impl AsRef<[u8]> for Prediction<'_> {
    /// Borrows prediction bytes for APIs accepting `AsRef<[u8]>`.
    ///
    /// Input: borrowed prediction.
    /// Output: borrowed byte slice.
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}
