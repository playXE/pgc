use libc::*;
use std::sync::atomic::{AtomicBool, AtomicI8, Ordering};
pub struct StkRoot {
    pub thread_handle: *mut u8,
    pub thread_id: u64,
    pub suspend_ack: AtomicBool,
}

unsafe impl Send for StkRoot {}
unsafe impl Sync for StkRoot {}

static mut GC_INSIDE_COLLECT: AtomicI8 = AtomicI8::new(-1);

use std::cell::RefCell;
use std::sync::Arc;

const GC_SIG_SUSPEND: i32 = SIGPWR;

thread_local! {
    pub static THREAD: RefCell<Arc<StkRoot>> = unsafe {

        let thread = StkRoot {
            thread_handle: std::ptr::null_mut(),
            suspend_ack: AtomicBool::new(false),
            thread_id: pthread_self()
        };
        let r = Arc::new(thread);
        THREADS.write().push(r.clone());
        RefCell::new(r)
    };
}

const GC_YIELD_MAX_ATTEMPT: usize = 2;

lazy_static::lazy_static! {
    pub static ref THREADS: parking_lot::RwLock<Vec<Arc<StkRoot>>> = parking_lot::RwLock::new(vec![]);
}

unsafe fn thread_yield(attempt: usize) {
    if attempt >= 2 {
        usleep((attempt - GC_YIELD_MAX_ATTEMPT) as u32 * 1000);
    } else {
        std::thread::yield_now();
    }
}

unsafe extern "C" fn suspend_handler(_: i32) {
    let mut attempt = 0;
    THREAD.with(|t| t.borrow_mut().suspend_ack.store(false, Ordering::Relaxed));
    loop {
        thread_yield(attempt);
        attempt += 1;
        if GC_INSIDE_COLLECT.load(Ordering::Relaxed) <= 0 {
            break;
        }
    }
}

pub unsafe fn mutator_suspend() {
    GC_INSIDE_COLLECT.store(1, Ordering::Relaxed);
    let lock = THREADS.read();
    for root in lock.iter() {
        if !THREAD.with(|t| Arc::ptr_eq(root, &t.borrow())) {
            root.suspend_ack.store(true, Ordering::Relaxed);
            signal(GC_SIG_SUSPEND, suspend_handler as usize);
            if pthread_kill(root.thread_id, GC_SIG_SUSPEND) != 0 {
                panic!("GC: Cannot send signal to thread: 0x{:x}", root.thread_id);
            }
        }
    }
    for root in lock.iter() {
        let mut attempt = 0;
        while root.suspend_ack.load(Ordering::Relaxed) {
            thread_yield(attempt);
            attempt += 1;
        }
    }
}

pub unsafe fn mutator_resume() {
    GC_INSIDE_COLLECT.store(0, Ordering::Relaxed);
}
