use crate::*;

#[test]
fn simple() {
    let foo = Rooted::new(vec![]);
    foo.get_mut().push(Gc::new(String::from("Hello,World!")));
    gc_collect();
    assert_eq!(foo.get().len(), 1);
    let val = foo.get_mut().pop().unwrap();
    gc_collect();
    assert!(!val.is_live());
}

#[test]
fn cyclic() {
    enum Foo {
        None,
        Node(i32, Gc<Foo>),
    }

    unsafe impl GcObject for Foo {
        fn references(&self) -> Vec<Gc<dyn GcObject>> {
            let mut v: Vec<Gc<dyn GcObject>> = vec![];
            match self {
                Foo::Node(_, n) => v.push(*n),
                _ => (),
            }
            v
        }
    }

    let v = Rooted::new(Foo::Node(0, Gc::new(Foo::None)));
    match v.get_mut() {
        Foo::Node(_, n) => *n = v.inner(),
        _ => (),
    }
    gc_collect();
    assert!(match v.get() {
        Foo::Node(_, n) => n.ref_eq(v.inner()),
        _ => false,
    });
    assert!(v.inner().is_live());
}
