#[cfg(feature = "ariel-os")]
mod ariel_os;

#[cfg(feature = "ariel-os")]
pub use ariel_os::ArielOsExecutor;

use super::{
    iree_hal_executable_dispatch_state_v0_t, iree_hal_executable_environment_v0_t,
    iree_hal_executable_workgroup_state_v0_t, DispatchFn,
};
use portable_atomic::{AtomicI32, Ordering};

#[derive(Clone, Copy)]
pub struct WorkItem {
    kind: WorkItemKind,
}

#[derive(Clone, Copy)]
enum WorkItemKind {
    IREEWorkload {
        dispatch_fn: DispatchFn,
        environment: *mut iree_hal_executable_environment_v0_t,
        dispatch_state: *mut iree_hal_executable_dispatch_state_v0_t,
        workgroup_state: iree_hal_executable_workgroup_state_v0_t,
        status: *const AtomicI32,
    },
}

impl WorkItem {
    /// Creates a work item borrowing dispatch state owned by the caller.
    ///
    /// # Safety
    ///
    /// All pointers must remain valid until the executor reports completion.
    pub(crate) unsafe fn iree(
        dispatch_fn: DispatchFn,
        environment: *mut iree_hal_executable_environment_v0_t,
        dispatch_state: *mut iree_hal_executable_dispatch_state_v0_t,
        workgroup_state: iree_hal_executable_workgroup_state_v0_t,
        status: *const AtomicI32,
    ) -> Self {
        Self {
            kind: WorkItemKind::IREEWorkload {
                dispatch_fn,
                environment,
                dispatch_state,
                workgroup_state,
                status,
            },
        }
    }

    pub(crate) fn run(self) {
        match self.kind {
            WorkItemKind::IREEWorkload {
                dispatch_fn,
                environment,
                dispatch_state,
                workgroup_state,
                status,
            } => {
                let dispatch_status =
                    unsafe { dispatch_fn(environment, dispatch_state, &workgroup_state) };
                if dispatch_status != 0 {
                    let status = unsafe { &*status };
                    let _ = status.compare_exchange(
                        0,
                        dispatch_status,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
            }
        }
    }
}

/// Schedules work items and provides a completion barrier.
pub trait Executor {
    /// Schedules and executes one work item.
    ///
    /// Input: work item to run.
    /// Output: result returned by the work item.
    fn schedule(&mut self, item: WorkItem);

    /// Waits until all previously scheduled work has finished.
    fn wait_job_completion(&mut self);
}

/// Default executor that runs work items immediately in submission order.
#[derive(Debug, Default, Clone, Copy)]
pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Creates a sequential executor.
    ///
    /// Input: none.
    /// Output: executor instance with no internal state.
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for SequentialExecutor {
    /// Runs the work item immediately on the current thread.
    ///
    /// Input: work item to execute.
    /// Output: result returned by the work item.
    fn schedule(&mut self, item: WorkItem) {
        item.run();
    }

    fn wait_job_completion(&mut self) {}
}

#[cfg(feature = "ariel-os")]
pub type DefaultExecutor = ArielOsExecutor;

#[cfg(not(feature = "ariel-os"))]
pub type DefaultExecutor = SequentialExecutor;
