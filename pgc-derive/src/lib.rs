extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate synstructure;
#[macro_use]
extern crate quote;
extern crate proc_macro2;

decl_derive!([GcObject, attributes(unsafe_ignore_trace)] => derive_gcobject);

fn derive_gcobject(mut s: synstructure::Structure) -> proc_macro2::TokenStream {
    s.filter(|bi| {
        !bi.ast()
            .attrs
            .iter()
            .any(|attr| attr.path.is_ident("unsafe_ignore_trace"))
    });
    let body = s.each(|bi| quote!(v.extend(#bi.references())));
    let gcobject_impl = s.unsafe_bound_impl(
        quote!(::pgc::GcObject),
        quote! {
            #[allow(unused_mut)]
            fn references(&self) -> Vec<pgc::Gc<dyn pgc::GcObject>> {
                let mut v: Vec<pgc::Gc<dyn pgc::GcObject>> = vec![];
                match *self {
                    #body
                };
                v
            }
        },
    );
    quote! {
        #gcobject_impl
    }
}
