use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use ariel_os::thread::sync::Channel;

use portable_atomic::{AtomicUsize, Ordering};

static JOB_REMAINING: AtomicUsize = AtomicUsize::new(0);

use super::{Executor, WorkItem};

use ariel_os::log::{debug, error, trace};

const TASK_SLOT_COUNT: usize = 4;
const TASK_SLOT_FREE: usize = 0;
const TASK_SLOT_BUSY: usize = 1;

#[derive(Clone, Copy)]
struct TaskMessage {
    item: WorkItem,
}

struct TaskSlot {
    state: AtomicUsize,
    message: UnsafeCell<MaybeUninit<TaskMessage>>,
}

unsafe impl Sync for TaskSlot {}

impl TaskSlot {
    const fn new() -> Self {
        Self {
            state: AtomicUsize::new(TASK_SLOT_FREE),
            message: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    fn write(&self, task: TaskMessage) {
        unsafe {
            (*self.message.get()).write(task);
        }
    }

    fn read(&self) -> TaskMessage {
        unsafe { (*self.message.get()).assume_init_read() }
    }
}

static TASKS: Channel<usize> = Channel::new();
static TASK_SLOTS: [TaskSlot; TASK_SLOT_COUNT] = [
    TaskSlot::new(),
    TaskSlot::new(),
    TaskSlot::new(),
    TaskSlot::new(),
];

fn acquire_task_slot() -> usize {
    loop {
        for (id, slot) in TASK_SLOTS.iter().enumerate() {
            if slot
                .state
                .compare_exchange(
                    TASK_SLOT_FREE,
                    TASK_SLOT_BUSY,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return id;
            }
        }

        ariel_os::thread::yield_same();
    }
}

fn release_task_slot(id: usize) {
    TASK_SLOTS[id]
        .state
        .store(TASK_SLOT_FREE, Ordering::Release);
}

/// Ariel OS executor backed by a small fixed worker pool.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArielOsExecutor;

impl ArielOsExecutor {
    /// Creates an Ariel OS executor handle.
    ///
    /// Input: none.
    /// Output: stateless executor handle using the global worker pool.
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for ArielOsExecutor {
    /// Runs the work item on an Ariel OS worker thread.
    ///
    /// Input: work item to execute.
    /// Output: result returned by the work item.
    fn schedule(&mut self, item: WorkItem) {
        let task = TaskMessage { item };
        let slot_id = acquire_task_slot();

        if slot_id >= TASK_SLOT_COUNT {
            error!("Scheduler acquired invalid task slot: {}", slot_id);
            panic!("Invalid task slot");
        }

        TASK_SLOTS[slot_id].write(task);

        JOB_REMAINING.fetch_add(1, Ordering::Relaxed);
        let my_id = ariel_os::thread::current_tid().unwrap();
        trace!(
            "[{:?}] Scheduling task: slot={}, item={:?}",
            my_id,
            slot_id,
            &task.item as *const WorkItem,
        );
        TASKS.send(&slot_id);
        trace!(
            "[{:?}] Finished scheduling task: slot={}, slot_id_addr={:?}",
            my_id,
            slot_id,
            &slot_id as *const usize
        );
    }

    fn wait_job_completion(&mut self) {
        while JOB_REMAINING.load(Ordering::Acquire) > 0 {
            ariel_os::thread::yield_same();
        }
    }
}

fn worker_loop() {
    let my_id = ariel_os::thread::current_tid().unwrap();
    let core = ariel_os::thread::core_id();
    debug!("[{:?}] Running on core {:?}", my_id, core);
    loop {
        trace!("[{:?}] Waiting for task", my_id);
        let slot_id = TASKS.recv();
        trace!(
            "[{:?}] Worker received task: slot={}, slot_id_addr={:?}",
            my_id,
            slot_id,
            &slot_id as *const usize
        );

        if slot_id >= TASK_SLOT_COUNT {
            error!(
                "[{:?}] Worker received invalid task slot: {}",
                my_id, slot_id
            );
            panic!("[{:?}] Invalid task slot", my_id);
        }

        let task = TASK_SLOTS[slot_id].read();
        trace!(
            "[{:?}] Worker running task: slot={}, item={:?}",
            my_id,
            slot_id,
            &task.item as *const WorkItem,
        );
        task.item.run();
        trace!(
            "[{:?}] Worker finished task: slot={}, slot_id_addr={:?}",
            my_id,
            slot_id,
            &slot_id as *const usize
        );
        release_task_slot(slot_id);
        JOB_REMAINING.fetch_sub(1, Ordering::Release);
    }
}

#[ariel_os::thread(autostart)]
fn oneliner_ariel_os_worker_0() {
    worker_loop();
}

#[ariel_os::thread(autostart)]
fn oneliner_ariel_os_worker_1() {
    worker_loop();
}
