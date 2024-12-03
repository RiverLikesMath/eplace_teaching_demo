#![allow(dead_code)]

use crate::util::calc_w;
use ndarray::Array2;

///Reference coefficient calculation and direct implementation of equation 21. Much slower than an fft library,
/// we implemented it while learning how to use fft/dct libraries and as a reference to be checked again. Calculates
/// one coefficient, a_u_v.
fn ref_dct_auv(density: &Array2<f64>, m: usize, u: usize, v: usize) -> f64 {
    let scale_factor = 1. / ((m as f64).powi(2)); //1/m^2

    let mut coefficient: f64 = 0.0; // a_u,v
    for x in 0..m {
        for y in 0..m {
            coefficient += scale_factor
                * density[[x, y]]
                * (calc_w(u, m) * (x as f64)).cos()
                * (calc_w(v, m) * (y as f64)).cos();
        }
    }
    coefficient
}

/// While the previous function calculated one coefficient, this calculates the entire mxm matrix of coefficients.
pub fn ref_dct_coeff(density: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut coeffs = Array2::<f64>::zeros((m, m));
    for u in 0..m {
        for v in 0..m {
            coeffs[[u, v]] = ref_dct_auv(density, m, u, v);
        }
    }

    //  coeffs.mapv_inplace(|x| x - dc);
    coeffs
}

///Calculate the density using an inverse cosign transform given a set of coefficients, up to a given scaling which might
/// not be correct at the moment! Reference version, directly implementing equation 22 from the paper, much slower than
/// if we used a DCT or FFT library.
pub fn ref_dct(coefficients: &Array2<f64>, m: usize, x: f64, y: f64) -> f64 {
    let mut density_dct = 0.;

    for u in 0..m {
        for v in 0..m {
            density_dct += coefficients[[u, v]]
                * (calc_w(u, m) * x).cos()
                * (calc_w(v, m) * y).cos();
        }
    }

    //    dbg!(&density);
    density_dct
}

///Calculates the matrix of x values of the electric field, form equation 24, first half, using the slow method.           
pub fn ref_elec_field_x(coeffs: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut elec_field_x = Array2::<f64>::zeros((m, m));
    for x in 0..m {
        for y in 0..m {
            elec_field_x[[x, y]] = calc_elec_point(coeffs, m, x, y);
        }
    }
    elec_field_x
}

/// calculate the electric field in the x direction using the slow (DSCT) algorithm in the paper for one
/// point (x,y)
fn calc_elec_point(coefficients: &Array2<f64>, m: usize, x: usize, y: usize) -> f64 {
    let mut field_x_at_point = 0.;
    for u in 0..m {
        for v in 0..m {
            let w_u = calc_w(u, m);
            let w_v = calc_w(v, m);

            if u == 0 && v == 0
            //formula as written has a divide by zero here
            {
                field_x_at_point += 0.;
            } else {
                let elec_x_coeff = coefficients[[u, v]] * w_u / (w_u.powi(2) + w_v.powi(2));

                let w_ux = w_u * (x as f64); //w_u * x
                let w_vy = w_v * (y as f64); //w_v * y

                field_x_at_point += elec_x_coeff * w_ux.sin() * w_vy.cos(); //eq 24,
                                                                            //the x
                                                                            //part
            }
        }
    }
    field_x_at_point
}
