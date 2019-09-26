use libc::*;
use std::sync::atomic::{AtomicBool, AtomicI8, Ordering};
pub struct StkRoot {
    pub thread_handle: *mut u8,
    pub thread_id: usize,
    pub suspend_ack: AtomicBool,
}

unsafe impl Send for StkRoot {}
unsafe impl Sync for StkRoot {}

static mut GC_INSIDE_COLLECT: AtomicI8 = AtomicI8::new(-1);

use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    pub static THREAD: RefCell<Arc<StkRoot>> = unsafe {

        let mut thread = StkRoot {
            thread_handle: std::ptr::null_mut(),
            suspend_ack: AtomicBool::new(false),
            thread_id: pthread_self()
        };
        let r = Arc::new(thread);
        THREADS.lock().push(r.clone());
        RefCell::new(r)
    };
}

lazy_static::lazy_static! {
    static ref THREADS: parking_lot::Mutex<Vec<Arc<StkRoot>>> = parking_lot::Mutex::new(vec![]);
}

struct GcMutexS {
    state: i64,
    event: *mut u8,
}
static mut GC_ALLOCATE: GcMutexS = GcMutexS {
    state: 0,
    event: std::ptr::null_mut(),
};
pub unsafe fn gc_enter() {}

pub unsafe fn mutator_suspend() {
    GC_INSIDE_COLLECT.store(1, Ordering::Relaxed);
}

pub unsafe fn mutator_resume() {}
