use mlx_rs::{
    array,
    macros::ModuleParameters,
    module::{ModuleParameters, Param, Parameter},
    test_utils::{assert_array_eq, tolerances},
    Array,
};

#[derive(ModuleParameters)]
pub struct StructModule {
    #[param]
    a: Param<Array>,

    #[param]
    b: Param<Array>,

    #[param]
    c: Param<Option<Array>>,
}

#[derive(ModuleParameters)]
pub struct UnitStructModule;

#[derive(ModuleParameters)]
struct NestedStructModule {
    #[param]
    a: Param<Array>,

    #[param]
    nested: StructModule,

    #[param]
    nested_no_param: UnitStructModule,
}

#[test]
fn test_module_num_parameters() {
    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    assert_eq!(m.num_parameters(), 2);

    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(Some(array!(3.0))),
    };

    assert_eq!(m.num_parameters(), 3);
}

#[test]
fn test_module_parameters() {
    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(Some(array!(3.0))),
    };

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["c"],
        &array!(3.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_module_parameters_mut() {
    let mut m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 2);
    assert_array_eq(
        &**flattened.get("a").unwrap(),
        &mut array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        &**flattened.get("b").unwrap(),
        &mut array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    let mut m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(Some(array!(3.0))),
    };

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 3);
    assert_array_eq(
        &**flattened.get("a").unwrap(),
        &mut array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        &**flattened.get("b").unwrap(),
        &mut array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        &**flattened.get("c").unwrap(),
        &mut array!(3.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_module_trainable_parameters_all_trainable() {
    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(Some(array!(3.0))),
    };

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["c"],
        &array!(3.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_module_trainable_parameters_partial_freeze() {
    let mut m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    // Freeze one parameter that is not optional
    m.a.freeze(true);

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 1);
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    // Now freeze the optional parameter
    m.c.freeze(true);

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 1);
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    // Unfreeze the non-optional parameter
    m.a.unfreeze(true);

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    // Set the optional parameter to Some but still frozen
    m.c.value = Some(array!(3.0));

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    // Unfreeze the optional parameter
    m.c.unfreeze(true);

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["b"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["c"],
        &array!(3.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_unit_struct_module_num_parameters() {
    let m = UnitStructModule;

    assert_eq!(m.num_parameters(), 0);
}

#[test]
fn test_unit_struct_module_parameters() {
    let m = UnitStructModule;

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 0);
}

#[test]
fn test_unit_struct_module_parameters_mut() {
    let mut m = UnitStructModule;

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 0);
}

#[test]
fn test_unit_struct_module_trainable_parameters() {
    let m = UnitStructModule;

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 0);
}

#[test]
fn test_unit_struct_module_freeze_parameters() {
    let mut m = UnitStructModule;

    m.freeze_parameters(true);
    assert_eq!(m.all_frozen(), None);
    assert_eq!(m.any_frozen(), None);
    assert_eq!(m.is_frozen(), None);
}

#[test]
fn test_unit_struct_module_unfreeze_parameters() {
    let mut m = UnitStructModule;

    m.unfreeze_parameters(true);
    assert_eq!(m.all_frozen(), None);
    assert_eq!(m.any_frozen(), None);
    assert_eq!(m.is_frozen(), None);
}

#[test]
fn test_nested_module_num_parameters() {
    let m = NestedStructModule {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
        nested_no_param: UnitStructModule,
    };
    assert_eq!(m.num_parameters(), 3);

    let m = NestedStructModule {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(Some(array!(4.0))),
        },
        nested_no_param: UnitStructModule,
    };
    assert_eq!(m.num_parameters(), 4);
}

#[test]
fn test_nested_module_parameters() {
    let m = NestedStructModule {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
        nested_no_param: UnitStructModule,
    };

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["nested.a"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["nested.b"],
        &array!(3.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_nested_module_parameters_mut() {
    let mut m = NestedStructModule {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
        nested_no_param: UnitStructModule,
    };

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 3);
    assert_array_eq(
        &**flattened.get("a").unwrap(),
        &mut array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        &**flattened.get("nested.a").unwrap(),
        &mut array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        &**flattened.get("nested.b").unwrap(),
        &mut array!(3.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_nested_module_recursive_freeze() {
    let mut m = NestedStructModule {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
        nested_no_param: UnitStructModule,
    };

    m.freeze_parameters(true);
    assert_eq!(m.all_frozen(), Some(true));

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 0);
}

#[test]
fn test_nested_module_freeze_submodule() {
    let mut m = NestedStructModule {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
        nested_no_param: UnitStructModule,
    };

    m.nested.freeze_parameters(true);
    assert_eq!(m.nested.all_frozen(), Some(true));
    assert_eq!(m.any_frozen(), Some(true));
    assert_eq!(m.all_frozen(), Some(false));

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 1);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_nested_module_unfreeze_submodule() {
    let mut m = NestedStructModule {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
        nested_no_param: UnitStructModule,
    };

    m.nested.freeze_parameters(true);
    m.nested.unfreeze_parameters(true);

    assert_eq!(m.any_frozen(), Some(false));

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["nested.a"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["nested.b"],
        &array!(3.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_nested_module_recursive_unfreeze() {
    let mut m = NestedStructModule {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
        nested_no_param: UnitStructModule,
    };

    m.freeze_parameters(true);
    m.unfreeze_parameters(true);
    assert_eq!(m.all_frozen(), Some(false));

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_array_eq(
        flattened["a"],
        &array!(1.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["nested.a"],
        &array!(2.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_array_eq(
        flattened["nested.b"],
        &array!(3.0),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}
