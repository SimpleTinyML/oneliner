#![no_std]

use oneliner_iree_abi::{
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
    IreeWorkload {
        dispatch_fn: DispatchFn,
        environment: *mut iree_hal_executable_environment_v0_t,
        dispatch_state: *mut iree_hal_executable_dispatch_state_v0_t,
        workgroup_state: iree_hal_executable_workgroup_state_v0_t,
        status: *const AtomicI32,
    },
}

impl WorkItem {
    /// Creates an IREE work item borrowing dispatch state owned by the caller.
    ///
    /// # Safety
    ///
    /// All pointers must remain valid until the executor reports completion.
    #[doc(hidden)]
    pub unsafe fn iree(
        dispatch_fn: DispatchFn,
        environment: *mut iree_hal_executable_environment_v0_t,
        dispatch_state: *mut iree_hal_executable_dispatch_state_v0_t,
        workgroup_state: iree_hal_executable_workgroup_state_v0_t,
        status: *const AtomicI32,
    ) -> Self {
        Self {
            kind: WorkItemKind::IreeWorkload {
                dispatch_fn,
                environment,
                dispatch_state,
                workgroup_state,
                status,
            },
        }
    }

    #[doc(hidden)]
    pub fn run(self) {
        match self.kind {
            WorkItemKind::IreeWorkload {
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
    /// Schedules one work item for execution.
    fn schedule(&mut self, item: WorkItem);

    /// Waits until all previously scheduled work has finished.
    fn wait_job_completion(&mut self);
}

/// Executor that runs work items immediately in submission order.
#[derive(Debug, Default, Clone, Copy)]
pub struct SequentialExecutor;

impl SequentialExecutor {
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for SequentialExecutor {
    fn schedule(&mut self, item: WorkItem) {
        item.run();
    }

    fn wait_job_completion(&mut self) {}
}
