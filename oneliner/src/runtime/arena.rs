#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

/// Instance-owned model arena storage selected by the OneLiner `alloc` feature.
pub struct OwnedArena<T> {
    #[cfg(not(feature = "alloc"))]
    inner: T,
    #[cfg(feature = "alloc")]
    inner: Box<T>,
}

impl<T> OwnedArena<T> {
    pub fn new(value: T) -> Self {
        Self {
            #[cfg(not(feature = "alloc"))]
            inner: value,
            #[cfg(feature = "alloc")]
            inner: Box::new(value),
        }
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T: Default> Default for OwnedArena<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

/// Only stores the large arena value.
///
/// If `value` is all-zero, this object can be placed in `.bss`.
pub struct ArenaStorage<T> {
    val: UnsafeCell<T>,
}

impl<T> ArenaStorage<T> {
    pub const fn new(value: T) -> Self {
        Self {
            val: UnsafeCell::new(value),
        }
    }
}

// Access is synchronized by the corresponding SharedArena.
unsafe impl<T: Send> Sync for ArenaStorage<T> {}

/// Synchronizes access to an arena shared by every instance of one model type.
#[cfg(feature = "ariel-os")]
pub struct SharedArena<T: 'static> {
    storage: &'static ArenaStorage<T>,
    lock: ariel_os::thread::sync::Lock,
}

#[cfg(feature = "ariel-os")]
impl<T: 'static> SharedArena<T> {
    pub const fn new(value: &'static ArenaStorage<T>) -> Self {
        Self {
            storage: value,
            lock: ariel_os::thread::sync::Lock::new(),
        }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        self.lock.acquire();

        let mut arena = unsafe { &mut *self.storage.val.get() };
        let res = f(&mut arena);
        self.lock.release();
        res
    }
}

/// Pure `no_std` fallback for targets without an OS-provided blocking mutex.
#[cfg(not(feature = "ariel-os"))]
pub struct SharedArena<T: 'static> {
    storage: &'static ArenaStorage<T>,
    _dummy_mutex: critical_section::Mutex<core::cell::RefCell<u8>>,
}

#[cfg(not(feature = "ariel-os"))]
impl<T: 'static> SharedArena<T> {
    pub const fn new(value: &'static ArenaStorage<T>) -> Self {
        Self {
            storage: value,
            _dummy_mutex: critical_section::Mutex::new(core::cell::RefCell::new(42)),
        }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        critical_section::with(|cs| {
            // Keep this guard alive for the entire closure.
            // It detects recursive access to the same arena.
            let _borrow =
                self._dummy_mutex.borrow_ref_mut(cs);

            let arena: &mut T = unsafe {
                &mut *self.storage.val.get()
            };

            f(arena)
        })
    }
}
unsafe impl<T: Send> Send for SharedArena<T> {}
unsafe impl<T: Send> Sync for SharedArena<T> {}