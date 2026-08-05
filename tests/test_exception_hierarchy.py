from ph_spline import (
    ArcLengthInversionError,
    ArcLengthOutOfRangeError,
    ConstructionConvergenceError,
    ContinuitySpecificationError,
    ContinuityVerificationError,
    CubicPHSplineError,
    CubicPHSplineRuntimeError,
    CubicPHSplineValueError,
    DegeneratePointDataError,
    DiscontinuousDerivativeError,
    G2VerificationError,
    InsufficientPointDataError,
    InterpolationDomainError,
    InterpolationVerificationError,
    InvalidPointDataError,
    LengthResolutionError,
    LocalEditFailure,
    NonAdmissibleSegmentError,
    NonFiniteCoordinateError,
    NonRegularSplineError,
    NonSimplePointDataError,
    NumericalPrecisionError,
    ParameterOutOfRangeError,
    PHBSplineError,
    PHBSplineRuntimeError,
    PHBSplineValueError,
    PHSplineError,
    PHSplineRuntimeError,
    PHSplineValueError,
    ResourceLimitError,
    ReversalError,
    SplineConvergenceError,
    StaleHandleError,
    StaleLocationError,
    TransactionError,
    UndefinedPrincipalNormalError,
)


def test_concrete_exception_families_are_siblings():
    assert CubicPHSplineError.__bases__ == (PHSplineError,)
    assert PHBSplineError.__bases__ == (PHSplineError,)
    for concrete in (CubicPHSplineError, PHBSplineError):
        assert issubclass(concrete, PHSplineError)
    for left, right in (
        (CubicPHSplineError, PHBSplineError),
        (CubicPHSplineValueError, PHBSplineValueError),
        (CubicPHSplineRuntimeError, PHBSplineRuntimeError),
    ):
        assert not issubclass(left, right)
        assert not issubclass(right, left)


def test_value_and_runtime_branches_have_neutral_bases():
    for value_error in (CubicPHSplineValueError, PHBSplineValueError):
        assert issubclass(value_error, PHSplineValueError)
        assert issubclass(value_error, ValueError)
    for runtime_error in (CubicPHSplineRuntimeError, PHBSplineRuntimeError):
        assert issubclass(runtime_error, PHSplineRuntimeError)
        assert issubclass(runtime_error, RuntimeError)


def test_shared_leaf_exceptions_belong_to_both_sibling_families():
    shared_value_errors = (
        InvalidPointDataError,
        InsufficientPointDataError,
        NonFiniteCoordinateError,
        DegeneratePointDataError,
        ParameterOutOfRangeError,
        ArcLengthOutOfRangeError,
        UndefinedPrincipalNormalError,
    )
    shared_runtime_errors = (
        NonRegularSplineError,
        ArcLengthInversionError,
        LengthResolutionError,
        NumericalPrecisionError,
    )
    for error in (*shared_value_errors, *shared_runtime_errors):
        assert issubclass(error, CubicPHSplineError)
        assert issubclass(error, PHBSplineError)


def test_family_specific_leaf_exceptions_do_not_cross_families():
    cubic_only = (
        NonSimplePointDataError,
        ReversalError,
        InterpolationDomainError,
        SplineConvergenceError,
        NonAdmissibleSegmentError,
        G2VerificationError,
    )
    bspline_only = (
        ContinuitySpecificationError,
        DiscontinuousDerivativeError,
        StaleHandleError,
        StaleLocationError,
        ConstructionConvergenceError,
        InterpolationVerificationError,
        ContinuityVerificationError,
        LocalEditFailure,
        ResourceLimitError,
        TransactionError,
    )
    for error in cubic_only:
        assert issubclass(error, CubicPHSplineError)
        assert not issubclass(error, PHBSplineError)
    for error in bspline_only:
        assert issubclass(error, PHBSplineError)
        assert not issubclass(error, CubicPHSplineError)
