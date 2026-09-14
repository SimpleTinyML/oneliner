//! Workspace ownership and platform synchronization.

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

/// Per-instance workspace; placed in a box with the `alloc` feature.
///
/// The workspace is independent of other instances. Boxing requires an
/// global allocator; constructing a large value can still use temporary
/// stack space before it is moved into the box.

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

/// Holds static workspace storage accessed through a shared arena guard.
///
/// If `value` is all-zero, this object can be placed in `.bss`.
pub struct ArenaStorage<T> {
    val: UnsafeCell<T>,
}

impl<T> ArenaStorage<T> {
    /// Wraps an initialized workspace without allocating or acquiring a lock.
    pub const fn new(value: T) -> Self {
        Self {
            val: UnsafeCell::new(value),
        }
    }
}

// Access is synchronized by the corresponding SharedArena.
unsafe impl<T: Send> Sync for ArenaStorage<T> {}

/// Uses an Ariel OS lock to serialize access to static workspace.
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

    /// Borrows the workspace while holding the guard.
    ///
    /// # Panics
    ///
    /// Propagates a panic from the closure. Recursive access panics with the
    /// critical-section implementation; recursive Ariel OS locking is not supported.
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        self.lock.acquire();

        let mut arena = unsafe { &mut *self.storage.val.get() };
        let res = f(&mut arena);
        self.lock.release();
        res
    }
}

/// Runs a closure inside a platform critical section over a static workspace.
///
/// The application must provide a `critical-section` implementation appropriate
/// for its interrupt and multicore environment. The critical section covers the
/// entire closure, which can affect interrupt latency.
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

    /// Borrows the workspace while holding the platform guard.
    ///
    /// # Panics
    ///
    /// Propagates a panic from the closure. Recursive access panics with the
    /// critical-section implementation.
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        critical_section::with(|cs| {
            // Keep this guard alive for the entire closure.
            // It detects recursive access to the same arena.
            let _borrow = self._dummy_mutex.borrow_ref_mut(cs);

            let arena: &mut T = unsafe { &mut *self.storage.val.get() };

            f(arena)
        })
    }
}
unsafe impl<T: Send> Send for SharedArena<T> {}
unsafe impl<T: Send> Sync for SharedArena<T> {}
