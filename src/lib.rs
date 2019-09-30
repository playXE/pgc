//! Simple mark&sweep garbage collector that can work in multiple threads (multithreaded collection support is still W.I.P).
//!
//! Internally this GC implementation uses mimalloc as fast memory allocator.
//! Marking is done in sync or in parallel.
//! Parallel marking is usually slower,but the idea is that the longer you wait before doing GC work, the more time it will take to do it once you get around to it.
//! So if you do a bit of the work regularly, you are less likely to experience stop-the-world pauses.
//!
#![feature(coerce_unsized)]
#![feature(unsize)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use scoped_threadpool::Pool;

use std::marker::Unsize;
use std::ops::CoerceUnsized;
use std::sync::atomic::AtomicBool;

#[cfg(feature = "generational")]
use std::sync::atomic::AtomicU8;

use mimalloc::MiMalloc;

#[cfg(target_family = "windows")]
pub mod win;

#[cfg(target_family = "unix")]
pub mod pthread;

#[cfg(target_family = "unix")]
use pthread::*;

#[cfg(target_family = "windows")]
use win::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static mut INCREMENTAL: AtomicBool = AtomicBool::new(false);
static mut GC_STATS: AtomicBool = AtomicBool::new(false);
lazy_static::lazy_static!(
    static ref TIMER: parking_lot::RwLock<Timer> = parking_lot::RwLock::new(Timer::new(false));
);
#[cfg(test)]
mod tests;

pub fn enable_gc_stats() {
    unsafe {
        GC_STATS.store(true, Ordering::Relaxed);
        *TIMER.write() = Timer::new(true);
    }
}
pub fn disable_gc_stats() {
    unsafe {
        GC_STATS.store(false, Ordering::Relaxed);
    }
}
use std::sync::Arc;
/// All threads should be attached automatically, but you could call this function if you unsure.
pub fn gc_attach_current_thread() {
    THREAD.with(|thread| {
        let mut threads = THREADS.write();
        for threadx in threads.iter() {
            if Arc::ptr_eq(threadx, &thread.borrow()) {
                return;
            }
        }
        threads.push(thread.borrow().clone());
    });
}

/// This function should be called when your thread finished execution ( basically not needed ).
pub fn gc_detach_current_thread() {
    THREAD.with(|thread| {
        let mut threads = THREADS.write();
        for i in 0..threads.len() {
            if Arc::ptr_eq(&threads[i], &thread.borrow()) {
                threads.remove(i);
                return;
            }
        }
    })
}

pub fn gc_summary() {
    COLLECTOR.with(|gc| gc.summary());
}

pub fn enable_incremental() {
    unsafe {
        INCREMENTAL.store(true, Ordering::Relaxed);
    }
}

pub fn disable_incremental() {
    unsafe {
        INCREMENTAL.store(false, Ordering::Relaxed);
    }
}

pub unsafe trait GcObject: Send {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        vec![]
    }
}

impl<T: GcObject + ?Sized + Unsize<U>, U: GcObject + ?Sized> CoerceUnsized<Gc<U>> for Gc<T> {}

pub struct GcPtr<T: ?Sized + GcObject> {
    mark: AtomicBool,
    #[cfg(feature = "generational")]
    gen: AtomicU8,
    #[cfg(test)]
    live: bool,
    val: T,
}
/// A garbage-collected pointer type
pub struct Gc<T: ?Sized + GcObject> {
    ptr: *mut GcPtr<T>,
}

use std::sync::atomic::Ordering as A;

unsafe impl<T: ?Sized + GcObject> Send for Gc<T> {}
unsafe impl<T: ?Sized + GcObject + Sync> Sync for Gc<T> {}

impl<T: Sized + GcObject + 'static> Gc<T> {
    /// Constructs a new `Gc<T>` with the given value.
    ///
    /// # Collection
    ///
    /// This method could trigger a garbage collection.
    ///
    pub fn new(val: T) -> Self {
        let mut gc = COLLECTOR.with(|gc| gc);

        gc.allocate(val)
    }
}

#[cfg(feature = "generational")]
pub fn gc_collect() {
    COLLECTOR.with(|mut gc| {
        gc.gen_collecting = 0;
        gc.collect();
        gc.gen_collecting = 3;
    })
}

#[cfg(feature = "generational")]
pub fn gc_collect_young() {
    COLLECTOR.with(|mut gc| {
        gc.gen_collecting = 0;
        gc.collect();
        gc.gen_collecting = 3;
    })
}

#[cfg(feature = "generational")]
pub fn gc_collect_old() {
    COLLECTOR.with(|mut gc| {
        gc.gen_collecting = 2;
        gc.collect();
        gc.gen_collecting = 3;
    })
}

pub fn gc_total_allocated() -> usize {
    COLLECTOR.with(|gc| gc.total_allocated)
}

#[cfg(feature = "generational")]
pub fn gc_collect_medium() {
    COLLECTOR.with(|mut gc| {
        gc.gen_collecting = 1;
        gc.collect();
        gc.gen_collecting = 3;
    })
}

/// Trigger parallel marking to mark objects.
///
/// You should call this function periodically to make GC pauses faster.
pub fn gc_mark_parallel() {
    COLLECTOR.with(|gc| {
        let mut lock = gc.pool.lock();
        start_marking_parallel(&gc.roots.read(), &mut lock, 0);
    })
}

#[cfg(not(feature = "generational"))]
pub fn gc_collect() {
    COLLECTOR.with(|mut gc| gc.collect());
}
use std::ops::Deref;
impl<T: GcObject + ?Sized> Deref for Gc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T: ?Sized + GcObject> Gc<T> {
    /// Get reference to data
    pub fn get(&self) -> &T {
        unsafe { &(&*self.ptr).val }
    }
    /// Get mutable reference to data.
    /// WARNING: Should be used carefully since Rust does not allows two mutable pointers to same data.
    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut (&mut *self.ptr).val }
    }
    /// Compares two pointers
    pub fn ref_eq(&self, other: Gc<dyn GcObject>) -> bool {
        self.ptr as *const u8 == other.ptr as *const u8
    }
    #[cfg(test)]
    pub(crate) fn is_live(&self) -> bool {
        unsafe { (&*self.ptr).live }
    }

    fn visit(&self, mut f: impl FnMut(Gc<dyn GcObject>)) {
        self.get().references().iter().for_each(|x| f(*x));
    }

    #[cfg(feature = "generational")]
    fn generation(&self) -> u8 {
        unsafe {
            let inner = &*self.ptr;
            inner.gen.load(Ordering::Relaxed)
        }
    }

    #[cfg(feature = "generational")]
    fn increase_generation(&self) {
        unsafe {
            let inner = &*self.ptr;
            let x = inner.gen.load(Ordering::Relaxed);
            if x < 2 {
                inner.gen.store(x + 1, Ordering::Relaxed);
            }
        }
    }

    fn is_marked(&self) -> bool {
        unsafe { (&*self.ptr).mark.load(A::Relaxed) }
    }
    fn mark(&self) -> bool {
        unsafe {
            match (&*self.ptr).mark.compare_exchange(
                false,
                true,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(false) => true,
                _ => false,
            }
        }
    }
    fn unmark(&self) {
        unsafe {
            (&*self.ptr).mark.store(false, A::Relaxed);
        }
    }
}
impl<T: ?Sized + GcObject> Clone for Gc<T> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}
impl<T: ?Sized + GcObject> Copy for Gc<T> {}

struct Handle<T> {
    handle: parking_lot::Mutex<T>,
}

impl<T> Handle<T> {
    fn with<'a, U>(&'a self, mut f: impl FnMut(parking_lot::MutexGuard<'a, T>) -> U) -> U {
        f(self.handle.lock())
    }
}

impl<T> Drop for Handle<T> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::drop_in_place(&mut self.handle);
        }
    }
}

lazy_static::lazy_static! {
    static ref COLLECTOR: Handle<Collector> = Handle {
        handle: parking_lot::Mutex::new(Collector::new())
    };
}

pub fn add_root(val: Gc<dyn GcObject>) {
    COLLECTOR.with(|gc| {
        let mut lock = gc.roots.write();
        for root in lock.iter() {
            if root.ptr == val.ptr {
                return;
            }
        }
        lock.push(val);
    });
}
pub fn remove_root(val: Gc<dyn GcObject>) {
    COLLECTOR.with(|gc| {
        let mut roots = gc.roots.write();
        for i in 0..roots.len() {
            if roots[i].ptr == val.ptr {
                roots.remove(i);
                return;
            }
        }
    });
}

pub fn is_root(val: Gc<dyn GcObject>) -> bool {
    let res = COLLECTOR.with(|gc| {
        for i in 0..gc.roots.read().len() {
            if gc.roots.read()[i].ptr == val.ptr {
                return true;
            }
        }
        false
    });
    res
}

struct CollectionStats {
    collections: usize,
    total_pause: f32,
    pauses: Vec<f32>,
}

pub struct AllNumbers(Vec<f32>);

impl fmt::Display for AllNumbers {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        let mut first = true;
        for num in &self.0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "{:.1}", num)?;
            first = false;
        }
        write!(f, "]")
    }
}

impl CollectionStats {
    fn new() -> CollectionStats {
        CollectionStats {
            collections: 0,
            total_pause: 0f32,
            pauses: Vec::new(),
        }
    }

    fn add(&mut self, pause: f32) {
        self.collections += 1;
        self.total_pause += pause;
        self.pauses.push(pause);
    }

    fn pause(&self) -> f32 {
        self.total_pause
    }

    fn pauses(&self) -> AllNumbers {
        AllNumbers(self.pauses.clone())
    }

    fn mutator(&self, runtime: f32) -> f32 {
        runtime - self.total_pause
    }

    fn collections(&self) -> usize {
        self.collections
    }

    fn percentage(&self, runtime: f32) -> (f32, f32) {
        let gc_percentage = ((self.total_pause / runtime) * 100.0).round();
        let mutator_percentage = 100.0 - gc_percentage;

        (mutator_percentage, gc_percentage)
    }
}

pub struct Collector {
    roots: parking_lot::RwLock<Vec<Gc<dyn GcObject>>>,
    heap: Vec<Gc<dyn GcObject>>,
    pool: parking_lot::Mutex<Pool>,
    threshold: usize,
    total_allocated: usize,
    #[cfg(feature = "generational")]
    threshold_young: usize,
    #[cfg(feature = "generational")]
    threshold_medium: usize,
    #[cfg(feature = "generational")]
    threshold_old: usize,

    #[cfg(feature = "generational")]
    allocated_old: usize,
    #[cfg(feature = "generational")]
    allocated_medium: usize,
    #[cfg(feature = "generational")]
    allocated_young: usize,
    gen_collecting: u8,
    stats: parking_lot::Mutex<CollectionStats>,
}

impl Collector {
    pub fn new() -> Collector {
        Collector {
            roots: parking_lot::RwLock::new(vec![]),
            heap: vec![],
            pool: parking_lot::Mutex::new(Pool::new(4)),
            threshold: 100,
            total_allocated: 0,
            #[cfg(feature = "generational")]
            threshold_medium: 400,
            #[cfg(feature = "generational")]
            threshold_old: 1000,
            #[cfg(feature = "generational")]
            threshold_young: 100,
            #[cfg(feature = "generational")]
            allocated_medium: 0,
            #[cfg(feature = "generational")]
            allocated_old: 0,
            #[cfg(feature = "generational")]
            allocated_young: 0,
            gen_collecting: 0,
            stats: parking_lot::Mutex::new(CollectionStats::new()),
        }
    }

    #[cfg(feature = "generational")]
    fn young_gen_size(&self) -> usize {
        let mut sum = 0;
        for item in self.heap.iter() {
            if item.generation() == 0 {
                sum += std::mem::size_of_val(item);
            }
        }
        sum
    }
    #[cfg(feature = "generational")]
    fn old_gen_size(&self) -> usize {
        let mut sum = 0;
        for item in self.heap.iter() {
            if item.generation() == 2 {
                sum += std::mem::size_of_val(item);
            }
        }
        sum
    }

    #[cfg(feature = "generational")]
    fn medium_gen_size(&self) -> usize {
        let mut sum = 0;
        for item in self.heap.iter() {
            if item.generation() == 1 {
                sum += std::mem::size_of_val(item);
            }
        }
        sum
    }
    #[cfg(feature = "generational")]
    pub fn allocate<T: GcObject + Sized + 'static>(&mut self, val: T) -> Gc<T> {
        unsafe {
            let layout = std::alloc::Layout::new::<GcPtr<T>>();
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                self.collect();
            }
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                panic!("OOM");
            }
            let ptr = ptr as *mut GcPtr<T>;
            ptr.write(GcPtr {
                mark: AtomicBool::new(false),
                #[cfg(feature = "generational")]
                gen: AtomicU8::new(0),
                #[cfg(test)]
                live: true,
                val,
            });
            mutator_suspend();
            if self.allocated_young > self.threshold_young {
                self.gen_collecting = 0;
                self.collect();
                if self.allocated_young as f64 > self.threshold_young as f64 * 0.7 {
                    self.threshold_young = (self.allocated_young as f64 / 0.7) as usize;
                }
            }
            if self.allocated_medium > self.threshold_medium {
                self.gen_collecting = 1;
                self.collect();
                if self.allocated_medium as f64 > self.threshold_medium as f64 * 0.7 {
                    self.threshold_medium = (self.allocated_medium as f64 / 0.7) as usize;
                }
            }
            if self.allocated_old > self.threshold_old {
                self.gen_collecting = 2;
                self.collect();
                if self.allocated_old as f64 > self.threshold_old as f64 * 0.7 {
                    self.threshold_old = (self.allocated_old as f64 / 0.7) as usize;
                }
            }
            self.allocated_young = self.allocated_young.wrapping_add(layout.size());

            let ptr = Gc {
                ptr: ptr as *mut GcPtr<T>,
            };
            self.gen_collecting = 3;
            self.heap.push(ptr);
            mutator_resume();
            ptr
        }
    }

    #[cfg(not(feature = "generational"))]
    pub fn allocate<T: GcObject + Sized + 'static>(&mut self, val: T) -> Gc<T> {
        unsafe {
            let layout = std::alloc::Layout::new::<GcPtr<T>>();
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                self.collect();
            }
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                panic!("OOM");
            }
            let ptr = ptr as *mut GcPtr<T>;
            ptr.write(GcPtr {
                mark: AtomicBool::new(false),
                #[cfg(feature = "generational")]
                gen: AtomicU8::new(0),
                #[cfg(test)]
                live: true,
                val,
            });
            mutator_suspend();
            /*if unsafe { INCREMENTAL.load(A::SeqCst) } {
                let rootset = self.roots.write();
                for root in rootset.iter() {
                    root.unmark();
                }
                let mut pool = self.pool.lock();
                start_marking_parallel(&rootset, &mut pool, 255);
            }*/
            if self.total_allocated > self.threshold {
                self.collect();
                if self.total_allocated as f64 > self.threshold as f64 * 0.7 {
                    self.threshold = (self.total_allocated as f64 / 0.7) as usize;
                }
            }
            self.total_allocated += layout.size();

            let ptr = Gc {
                ptr: ptr as *mut GcPtr<T>,
            };

            self.heap.push(ptr);
            mutator_resume();
            ptr
        }
    }

    pub fn collect(&mut self) {
        unsafe {
            mutator_suspend();
        }
        let mut timer = Timer::new(unsafe { GC_STATS.load(Ordering::Relaxed) });
        //if unsafe { INCREMENTAL.load(Ordering::Relaxed) } {
        /*if self.heap.len() < 100 {
            let mut stack = vec![];
            for root in self.roots.read().iter() {
                if root.mark() {
                    stack.push(*root);
                }
            }
            while let Some(object) = stack.pop() {
                object.visit(|x| {
                    if x.mark() {
                        stack.push(x);
                    }
                });
            }
        } else */
        /*    {
                let mut pool = self.pool.lock();
                let gen = self.gen_collecting;
                start_marking_parallel(&self.roots.read(), &mut pool, gen);
            }
        } else*/
        {
            // Sync marking.
            let mut stack = vec![];
            for root in self.roots.read().iter() {
                if root.mark() {
                    stack.push(*root);
                }
            }
            while let Some(object) = stack.pop() {
                object.visit(|x| {
                    if x.mark() {
                        stack.push(x);
                    }
                });
            }
        }
        let mut heap = vec![];
        for item in self.heap.iter() {
            if item.is_marked() {
                item.unmark();
                #[cfg(feature = "generational")]
                {
                    item.increase_generation();
                    let gen = item.generation();
                    match gen {
                        1 => self.allocated_medium += std::mem::size_of_val(item),
                        2 => self.allocated_old += std::mem::size_of_val(item),
                        _ => unreachable!(),
                    };
                    match gen - 1 {
                        0 => {
                            self.allocated_young = self
                                .allocated_medium
                                .wrapping_sub(std::mem::size_of_val(item))
                        }
                        1 => {
                            self.allocated_medium = self
                                .allocated_medium
                                .wrapping_sub(std::mem::size_of_val(item))
                        }
                        _ => unreachable!(),
                    };
                }
                heap.push(*item);
            } else {
                #[cfg(test)]
                unsafe {
                    (&mut *item.ptr).live = false;
                }
                #[cfg(feature = "generational")]
                {
                    if item.generation() == self.gen_collecting {
                        if self.gen_collecting == 0 {
                            self.allocated_young = self
                                .allocated_young
                                .wrapping_sub(std::mem::size_of_val(item));
                        } else if self.gen_collecting == 1 {
                            self.allocated_medium = self
                                .allocated_medium
                                .wrapping_sub(std::mem::size_of_val(item));
                        } else if self.gen_collecting == 2 {
                            self.allocated_old =
                                self.allocated_old.wrapping_sub(std::mem::size_of_val(item));
                        }

                        unsafe {
                            let _ = Box::from_raw(item.ptr);
                        }
                    } else {
                        heap.push(*item);
                    }
                }
                #[cfg(not(feature = "generational"))]
                {
                    self.total_allocated = self
                        .total_allocated
                        .wrapping_sub(std::mem::size_of_val(item));

                    unsafe {
                        let _ = Box::from_raw(item.ptr);
                    }
                }
            }
        }
        self.heap = heap;
        #[cfg(feature = "generational")]
        {
            if self.allocated_medium > self.threshold_medium {
                self.gen_collecting = 1;
                self.collect();
                self.gen_collecting = 3;
                if self.allocated_medium as f64 > self.threshold_medium as f64 * 0.7 {
                    self.threshold_medium = (self.allocated_medium as f64 / 0.7) as usize;
                }
            }
            if self.allocated_old > self.threshold_old {
                self.gen_collecting = 2;
                self.collect();
                self.gen_collecting = 3;
                if self.allocated_old as f64 > self.threshold_old as f64 * 0.7 {
                    self.threshold_old = (self.allocated_old as f64 / 0.7) as usize;
                }
            }
        }

        unsafe {
            if GC_STATS.load(Ordering::Relaxed) {
                let duration = timer.stop();
                let mut stats = self.stats.lock();
                stats.add(duration);
            }
            mutator_resume();
        }
    }
    fn summary(&self) {
        let mut timer = TIMER.write();
        let runtime = timer.stop();
        let stats = self.stats.lock();

        let (mutator, gc) = stats.percentage(runtime);
        eprintln!("GC stats: total={:.1}", runtime);
        eprintln!("GC stats: mutator={:.1}", stats.mutator(runtime));
        eprintln!("GC stats: collection={:.1}", stats.pause());

        eprintln!("");
        eprintln!("GC stats: collection-count={}", stats.collections());
        eprintln!("GC stats: collection-pauses={}", stats.pauses());
        eprintln!("GC stats: threshold={}", self.threshold);
        eprintln!("GC stats: total allocated={}", self.total_allocated);

        eprintln!(
            "GC summary: {:.1}ms collection ({}), {:.1}ms mutator, {:.1}ms total ({}% mutator, {}% GC)",
            stats.pause(),
            stats.collections(),
            stats.mutator(runtime),
            runtime,
            mutator,
            gc,
        );
    }
}

impl Drop for Collector {
    fn drop(&mut self) {
        if unsafe { GC_STATS.load(Ordering::Relaxed) } {
            self.summary();
        }
        for item in self.heap.iter() {
            unsafe {
                let _ = Box::from_raw(item.ptr);
            }
        }
    }
}

fn start_marking_parallel(rootset: &[Gc<dyn GcObject>], threadpool: &mut Pool, _: u8) {
    let number_workers = threadpool.thread_count() as usize;
    let mut workers = Vec::with_capacity(number_workers);
    let mut stealers = Vec::with_capacity(number_workers);
    let injector = Injector::new();

    for _ in 0..number_workers {
        let w = Worker::new_lifo();
        let s = w.stealer();
        workers.push(w);
        stealers.push(s);
    }
    for root in rootset {
        // Object unmarked? Mark it and push to stack.
        if root.mark() {
            injector.push(root.clone());
        }
    }
    let terminator = Terminator::new(number_workers);

    threadpool.scoped(|scoped| {
        for (task_id, worker) in workers.into_iter().enumerate() {
            let injector = &injector;
            let stealers = &stealers;
            let terminator = &terminator;

            scoped.execute(move || {
                let mut task = MarkingTask {
                    task_id: task_id,
                    local: Segment::new(),
                    worker: worker,
                    injector: injector,
                    stealers: stealers,
                    terminator: terminator,
                    marked: 0,
                };

                task.run();
            });
        }
    });
}

pub struct Terminator {
    const_nworkers: usize,
    nworkers: AtomicUsize,
}

impl Terminator {
    pub fn new(number_workers: usize) -> Terminator {
        Terminator {
            const_nworkers: number_workers,
            nworkers: AtomicUsize::new(number_workers),
        }
    }

    pub fn try_terminate(&self) -> bool {
        if self.const_nworkers == 1 {
            return true;
        }

        self.decrease_workers();
        thread::sleep(Duration::from_micros(1));
        self.zero_or_increase_workers()
    }

    fn decrease_workers(&self) -> bool {
        self.nworkers.fetch_sub(1, Ordering::SeqCst) == 1
    }

    fn zero_or_increase_workers(&self) -> bool {
        let mut nworkers = self.nworkers.load(Ordering::Relaxed);

        loop {
            if nworkers == 0 {
                return true;
            }

            let prev_nworkers =
                self.nworkers
                    .compare_and_swap(nworkers, nworkers + 1, Ordering::SeqCst);

            if nworkers == prev_nworkers {
                return false;
            }

            nworkers = prev_nworkers;
        }
    }
}

struct MarkingTask<'a> {
    task_id: usize,
    local: Segment,
    worker: Worker<Gc<dyn GcObject>>,
    injector: &'a Injector<Gc<dyn GcObject>>,
    stealers: &'a [Stealer<Gc<dyn GcObject>>],
    terminator: &'a Terminator,
    marked: usize,
}

impl<'a> MarkingTask<'a> {
    fn pop(&mut self) -> Option<Gc<dyn GcObject>> {
        self.pop_local()
            .or_else(|| self.pop_worker())
            .or_else(|| self.pop_global())
            .or_else(|| self.steal())
    }

    fn pop_local(&mut self) -> Option<Gc<dyn GcObject>> {
        if self.local.is_empty() {
            return None;
        }

        let obj = self.local.pop().expect("should be non-empty");
        Some(obj)
    }

    fn pop_worker(&mut self) -> Option<Gc<dyn GcObject>> {
        self.worker.pop()
    }

    fn pop_global(&mut self) -> Option<Gc<dyn GcObject>> {
        loop {
            let result = self.injector.steal_batch_and_pop(&mut self.worker);

            match result {
                Steal::Empty => break,
                Steal::Success(value) => return Some(value),
                Steal::Retry => continue,
            }
        }

        None
    }

    fn steal(&self) -> Option<Gc<dyn GcObject>> {
        if self.stealers.len() == 1 {
            return None;
        }

        let mut rng = thread_rng();
        let range = Uniform::new(0, self.stealers.len());

        for _ in 0..2 * self.stealers.len() {
            let mut stealer_id = self.task_id;

            while stealer_id == self.task_id {
                stealer_id = range.sample(&mut rng);
            }

            let stealer = &self.stealers[stealer_id];

            loop {
                match stealer.steal_batch_and_pop(&self.worker) {
                    Steal::Empty => break,
                    Steal::Success(address) => return Some(address),
                    Steal::Retry => continue,
                }
            }
        }

        None
    }
    fn run(&mut self) {
        loop {
            let object = if let Some(object) = self.pop() {
                object
            } else if self.terminator.try_terminate() {
                break;
            } else {
                continue;
            };
            // Visit all references of object and mark them.
            object.visit(|field| {
                self.trace(field);
            });
        }
    }

    fn trace(&mut self, field: Gc<dyn GcObject>) {
        // if reference (field) is unmarked push it to stack and then visit it.
        if field.mark() {
            if self.local.has_capacity() {
                self.local.push(field);
                self.defensive_push();
            } else {
                self.worker.push(field);
            }
        }
    }
    fn defensive_push(&mut self) {
        self.marked += 1;

        if self.marked > 256 {
            if self.local.len() > 4 {
                let target_len = self.local.len() / 2;

                while self.local.len() > target_len {
                    let val = self.local.pop().unwrap();
                    self.injector.push(val);
                }
            }

            self.marked = 0;
        }
    }
}

const SEGMENT_SIZE: usize = 64;

struct Segment {
    data: Vec<Gc<dyn GcObject>>,
}

impl Segment {
    fn new() -> Segment {
        Segment {
            data: Vec::with_capacity(SEGMENT_SIZE),
        }
    }

    fn has_capacity(&self) -> bool {
        self.data.len() < SEGMENT_SIZE
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn push(&mut self, addr: Gc<dyn GcObject>) {
        debug_assert!(self.has_capacity());
        self.data.push(addr);
    }

    fn pop(&mut self) -> Option<Gc<dyn GcObject>> {
        self.data.pop()
    }

    fn len(&mut self) -> usize {
        self.data.len()
    }
}

unsafe impl<T: GcObject + 'static> GcObject for Gc<T> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut v: Vec<Gc<dyn GcObject>> = vec![];
        v.push(*self);
        self.visit(|x| {
            v.push(x);
        });
        v
    }
}

unsafe impl<T: GcObject> GcObject for Vec<T> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut refs: Vec<Gc<dyn GcObject>> = vec![];
        for item in self.iter() {
            for item in item.references() {
                refs.push(item);
            }
        }
        refs
    }
}

macro_rules! simple {
    ($($t: ty)*) => {
        $(
            unsafe impl GcObject for $t {}
        )*
    };
}

simple!(
    i8
    i16
    i32
    i64
    i128
    u8
    u16
    u32
    u64
    u128
    f64
    f32
    bool
    String
    isize
    usize
    std::fs::File
    std::fs::FileType
    std::fs::Metadata
    std::fs::OpenOptions
    std::io::Stdin
    std::io::Stdout
    std::io::Stderr
    std::io::Error
    std::net::TcpStream
    std::net::TcpListener
    std::net::UdpSocket
    std::net::Ipv4Addr
    std::net::Ipv6Addr
    std::net::SocketAddrV4
    std::net::SocketAddrV6
    std::path::Path
    std::path::PathBuf
    std::process::Command
    std::process::Child
    std::process::ChildStdout
    std::process::ChildStdin
    std::process::ChildStderr
    std::process::Output
    std::process::ExitStatus
    std::process::Stdio
    std::sync::Barrier
    std::sync::Condvar
    std::sync::Once
    std::ffi::CStr
    std::ffi::CString
    &'static str
);

macro_rules! trace_arrays {
    ($($N: expr),*) => {
        $(
            unsafe impl<T: GcObject> GcObject for [T;$N] {
                fn references(&self) -> Vec<Gc<dyn GcObject>> {
                    let mut refs: Vec<Gc<dyn GcObject>> = vec![];
                    for item in self.iter() {
                        for item in item.references() {
                            refs.push(item);
                        }
                    }
                    refs
                }
            }
        )*
    };
}

trace_arrays! {
    0o00, 0o01, 0o02, 0o03, 0o04, 0o05, 0o06, 0o07,
    0o10, 0o11, 0o12, 0o13, 0o14, 0o15, 0o16, 0o17,
    0o20, 0o21, 0o22, 0o23, 0o24, 0o25, 0o26, 0o27,
    0o30, 0o31, 0o32, 0o33, 0o34, 0o35, 0o36, 0o37
}

macro_rules! trace_tuples {
    ($(($($T:ident : $N:tt),*))*) => {
        $(
            unsafe impl<$($T: GcObject,)*> GcObject for ($($T,)*) {
                #[allow(unused_mut)]
                fn references(&self) -> Vec<Gc<dyn GcObject>> {
                    let mut refs: Vec<Gc<dyn GcObject>> = vec![];
                    $(
                        refs.extend(self.$N.references());
                    )*
                    refs
                }
            }
        )*
    };
}

trace_tuples! {
    ()
    (A: 0)
    (A: 0, B: 1)
    (A: 0, B: 1, C: 2)
    (A: 0, B: 1, C: 2, D: 3)
    (A: 0, B: 1, C: 2, D: 3, E: 4)
    (A: 0, B: 1, C: 2, D: 3, E: 4, F: 5)
    (A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6)
    (A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7)
    (A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8)
    (A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9)
    (A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10)
    (A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11)
}

use std::collections::*;

unsafe impl<T: GcObject> GcObject for VecDeque<T> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let refs = self.iter().map(|x| x.references()).flatten().collect();
        refs
    }
}

unsafe impl<T: GcObject> GcObject for LinkedList<T> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let refs = self.iter().map(|x| x.references()).flatten().collect();
        refs
    }
}

unsafe impl<K: GcObject, V: GcObject> GcObject for HashMap<K, V> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut refs: Vec<Gc<dyn GcObject>> = vec![];
        for (key, val) in self.iter() {
            for (key, val) in key.references().iter().zip(&val.references()) {
                refs.push(*key);
                refs.push(*val);
            }
        }
        refs
    }
}

unsafe impl<K: GcObject> GcObject for HashSet<K> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let refs = self.iter().map(|x| x.references()).flatten().collect();
        refs
    }
}

/// An structure to hold root object alive until end of it's live. Internally this structure uses reference counting.
///
/// ```rust
/// use ngc::*;
///
/// {
///     let object = Rooted::new(42); // is_root(object) = true
///
/// } // drop `Rooted` and unroot allocated object
/// gc_collect();
///
/// ```
pub struct Rooted<T: GcObject + 'static> {
    ptr: std::sync::Arc<Gc<T>>,
}

impl<T: 'static + GcObject> Clone for Rooted<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr.clone(),
        }
    }
}
impl<T: 'static + GcObject> Drop for Rooted<T> {
    fn drop(&mut self) {
        remove_root(*self.ptr);
    }
}

impl<T: GcObject + 'static> Rooted<T> {
    /// Construct new rooted object.
    pub fn new(val: T) -> Self {
        let val = Gc::new(val);
        add_root(val);
        Self {
            ptr: std::sync::Arc::new(val),
        }
    }

    pub fn with(x: Gc<T>) -> Self {
        let x = Self {
            ptr: std::sync::Arc::new(x),
        };
        add_root(*x.ptr);
        x
    }

    pub fn inner(&self) -> Gc<T> {
        *self.ptr
    }

    pub fn get_mut(&self) -> &mut T {
        self.ptr.get_mut()
    }
    pub fn get(&self) -> &T {
        self.ptr.get()
    }
}

impl<T: GcObject + 'static> From<Rooted<T>> for Gc<T> {
    fn from(x: Rooted<T>) -> Self {
        x.inner()
    }
}

use std::fmt;

impl<T: fmt::Debug + GcObject> fmt::Debug for Gc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.get())
    }
}

impl<T: fmt::Display + GcObject> fmt::Display for Gc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

impl<T: PartialEq + GcObject> PartialEq for Gc<T> {
    fn eq(&self, x: &Self) -> bool {
        *self.get() == *x.get()
    }
}

impl<T: Eq + PartialEq + GcObject> Eq for Gc<T> {}

impl<T: PartialOrd + GcObject> PartialOrd for Gc<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.get().partial_cmp(other.get())
    }
}
impl<T: PartialOrd + Ord + GcObject> Ord for Gc<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get().cmp(other.get())
    }
}

use std::hash::{Hash, Hasher};

impl<T: Hash + GcObject> Hash for Gc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get().hash(state);
    }
}

unsafe impl<T: GcObject> GcObject for Option<T> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut v: Vec<Gc<dyn GcObject>> = vec![];
        match self {
            Some(val) => v.extend(val.references()),
            _ => (),
        }
        v
    }
}

unsafe impl<T: GcObject> GcObject for parking_lot::Mutex<T> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut v: Vec<Gc<dyn GcObject>> = vec![];
        v.extend(self.lock().references());
        v
    }
}

unsafe impl<T: GcObject> GcObject for parking_lot::RwLock<T> {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut v: Vec<Gc<dyn GcObject>> = vec![];
        v.extend(self.read().references());
        v
    }
}

pub struct Timer {
    active: bool,
    timestamp: u64,
}

impl Timer {
    pub fn new(active: bool) -> Timer {
        let ts = if active { timestamp() } else { 0 };

        Timer {
            active: active,
            timestamp: ts,
        }
    }

    pub fn stop(&mut self) -> f32 {
        assert!(self.active);
        let curr = timestamp();
        let last = self.timestamp;
        self.timestamp = curr;

        in_ms(curr - last)
    }

    pub fn stop_with<F>(&self, f: F) -> u64
    where
        F: FnOnce(f32),
    {
        if self.active {
            let ts = timestamp() - self.timestamp;

            f(in_ms(ts));

            ts
        } else {
            0
        }
    }

    pub fn ms<F>(active: bool, f: F) -> f32
    where
        F: FnOnce(),
    {
        if active {
            let ts = timestamp();
            f();
            let diff = timestamp() - ts;
            in_ms(diff)
        } else {
            f();
            0.0f32
        }
    }
}

pub fn in_ms(ns: u64) -> f32 {
    (ns as f32) / 1000.0 / 1000.0
}

pub fn timestamp() -> u64 {
    time::precise_time_ns()
}
