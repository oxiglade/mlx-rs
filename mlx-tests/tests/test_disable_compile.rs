use mlx_rs::{
    array,
    error::Exception,
    ops::{exp, negative},
    transforms::compile::{compile, disable_compile, enable_compile},
    Array,
};

#[test]
fn test_disable_compile() {
    disable_compile();

    let f = |x: &Array| -> Result<Array, Exception> {
        let z = negative(x)?;

        // this will crash is compile is enabled
        println!("{z:?}");

        exp(z)
    };

    let x = array!(10.0);
    let mut compiled = compile(f, None);

    // This will panic if compilation is enabled
    let _result = compiled(&x).unwrap();

    // Re-enable compilation for other tests
    enable_compile();
}
