use mlx_rs::{
    array,
    error::IoError,
    macros::ModuleParameters,
    module::{ModuleParameters, ModuleParametersExt, Param, Parameter},
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
fn test_load_safetensors_validation_is_atomic() {
    let cases = [
        (
            vec![
                ("a", array!([10.0f32, 20.0])),
                ("b", array!([30.0f32, 40.0])),
                ("z", array!(50.0f32)),
                ("c", array!(60.0f32)),
            ],
            IoError::UnexpectedParameters(vec!["c".into(), "z".into()]),
        ),
        (
            vec![("a", array!([10.0f32, 20.0]))],
            IoError::MissingParameters(vec!["b".into()]),
        ),
        (
            vec![
                ("a", array!([10.0f32, 20.0])),
                ("b", array!([[30.0f32, 40.0]])),
            ],
            IoError::ParameterShapeMismatch {
                key: "b".into(),
                expected: vec![2],
                actual: vec![1, 2],
            },
        ),
    ];
    let tmp_dir = tempfile::tempdir().unwrap();
    let path = tmp_dir.path().join("module.safetensors");
    for (weights, expected_error) in cases {
        let mut module = StructModule {
            a: Param::new(array!([1.0f32, 2.0])),
            b: Param::new(array!([3.0f32, 4.0])),
            c: Param::new(None),
        };
        let original: Vec<_> = module
            .parameters()
            .flatten()
            .into_iter()
            .map(|(key, value)| (key, value.clone()))
            .collect();
        Array::save_safetensors(weights, None, &path).unwrap();

        assert_eq!(module.load_safetensors(&path), Err(expected_error));

        let params = module.parameters().flatten();
        assert_eq!(params.len(), original.len());
        for (key, value) in original {
            assert!(params[&key].eq_exact(value).unwrap());
        }
        assert!(module.c.value.is_none());
    }
}

#[test]
fn test_module_safetensors_roundtrip() {
    let source = NestedStructModule {
        a: Param::new(array!(10i32)),
        nested: StructModule {
            a: Param::new(array!([20i32, 30])),
            b: Param::new(array!([[40i32, 50]])),
            c: Param::new(Some(array!(60i32))),
        },
        nested_no_param: UnitStructModule,
    };
    let mut target = NestedStructModule {
        a: Param::new(array!(0.0f32)),
        nested: StructModule {
            a: Param::new(array!([0.0f32, 0.0])),
            b: Param::new(array!([[0.0f32, 0.0]])),
            c: Param::new(Some(array!(0.0f32))),
        },
        nested_no_param: UnitStructModule,
    };
    target.freeze_parameters(true);
    let tmp_dir = tempfile::tempdir().unwrap();
    let path = tmp_dir.path().join("module.safetensors");
    source.save_safetensors(&path).unwrap();

    target.load_safetensors(&path).unwrap();

    let expected = source.parameters().flatten();
    let actual = target.parameters().flatten();
    assert_eq!(actual.len(), expected.len());
    for (key, value) in expected {
        assert!(actual[&key].eq_exact(value).unwrap());
    }
    assert_eq!(target.all_frozen(), Some(true));
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
