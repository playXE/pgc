# PGC
Precise tracing garbage collector built in Rust featuring parallel marking and mark&sweep algorithm.


# How to use
To include in your project, add the following to your Cargo.toml:
```toml
[dependencies]
pgc = "*"
```
This can be used pretty much like Rc, with the exception of interior mutability.

Types placed inside a `Gc` and `Rooted` must implement `GcObject` and `Send`.
```rust
use pgc::*;

struct Foo {
    x: Gc<i32>,
    y: i32
}

unsafe impl GcObject for Foo {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut v: Vec<Gc<dyn GcObject>> = vec![];
        v.push(self.x);
        v
    }
}
```

To use `Gc` simply call `Gc::new`:
```rust
let foo = Gc::new(Foo {...});
```
GC does not scan program stack for root objects so you should add roots explicitly:
```rust
let foo = Gc::new(Foo {...});
add_root(foo);
... // do something with `foo`
remove_root(foo);
// Or use `Rooted` struct that will unroot object automatically:
let foo = Rooted:new(Foo {...});
```


# Issues
- ~~The current collection algorithm is not fully thread safe, for collecting objects in multiple threads we should provide some platform-dependent code for pausing threads (stop-the-world)~~ Seems that stop-the-world mechanism works.
- Not fully incremental. Marking can be done in parallel but if you want to mark small pieces of memory you should call `gc_mark` in your code.
- GC can't properly scan program stack for objects since Rust dynamic dispatch does not fully allow casting some random pointer to trait object and because of that you should root and unroot objects explicitly.