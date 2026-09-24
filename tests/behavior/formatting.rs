//! Diagnostic formatting propagates destination failures without changing prepared state.

use fgf::{Gf8B, gf8b};
use poly_ring::*;
use std::fmt::{self, Debug, Write};

struct Reject(usize);
impl Write for Reject {
    fn write_str(&mut self, _: &str) -> fmt::Result {
        if self.0 == 0 {
            return Err(fmt::Error);
        }
        self.0 -= 1;
        Ok(())
    }
}

fn rejects(value: &impl Debug) {
    let mut writes = Reject(usize::MAX);
    write!(writes, "{value:?}").unwrap();
    let count = usize::MAX - writes.0;
    for allowed in 0..count {
        assert_eq!(write!(Reject(allowed), "{value:?}"), Err(fmt::Error));
    }
}

#[test]
fn prepared_diagnostics_propagate_writer_errors() {
    let points = [gf8b::Elem::from_raw(1), gf8b::Elem::from_raw(2)];
    let polynomial = Polynomial::<Gf8B>::from_coefficients(&points).unwrap();
    rejects(&polynomial);
    let derivative = DerivativePlan::<Gf8B>::new(1, 2).unwrap();
    rejects(&derivative);
    let jet = JetPlan::<Gf8B>::new(points[0], 2, 2).unwrap();
    rejects(&jet);
    rejects(&jet.scratch(1).unwrap());
    let weighted = MultiplicityPlan::<Gf8B>::new(&points, &[1, 1], 2).unwrap();
    rejects(&weighted);
    rejects(&weighted.scratch(1).unwrap());
    let hermite = HermitePlan::<Gf8B>::new(&points, &[1, 1]).unwrap();
    rejects(&hermite);
    let tree = RemainderTree::new(std::slice::from_ref(&polynomial), 2).unwrap();
    rejects(&tree);
    rejects(&tree.scratch(1).unwrap());
    rejects(&ConvolutionScratch::<Gf8B>::new(2, 2, 1).unwrap());
    #[cfg(feature = "fft")]
    rejects(&PolynomialProductScratch::<Gf8B>::new());
    assert_eq!(polynomial.coefficients().collect::<Vec<_>>(), points);
}

#[test]
fn wrapped_error_diagnostics_propagate_writer_failure() {
    fn fails(error: impl std::fmt::Display) {
        assert_eq!(write!(Reject(0), "{error}"), Err(fmt::Error));
    }
    let config = ConfigError::GeometryOverflow { context: "request" };
    fails(RootError::Product(ProductError::Config(config)));
    fails(DomainError::Config(config));
    fails(EvalError::Polynomial(PolynomialError::DivisionByZero));
    fails(HermiteError::Config(config));
    fails(HermiteError::Polynomial(PolynomialError::DivisionByZero));
    fails(HasseError::Polynomial(PolynomialError::DivisionByZero));
}
