use ndarray::Array2;
use std::f64::consts::PI;

pub fn dct_coeff(density: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut coefficients = Array2::<f64>::zeros((m, m));
    for u in 0..m {
        for v in 0..m {
            coefficients[[u, v]] = eplace_dct_auv(density, m, u, v);
        }
    }
    coefficients
}
pub fn eplace_dct(coefficients: &Array2<f64>, m: usize, x: f64, y: f64) -> f64 {
    let mut density_dct = 0.;

    for u in 0..m {
        for v in 0..m {
            density_dct += coefficients[[u, v]]
                * (calc_w(u, m) * (x as f64)).cos()
                * (calc_w(v, m) * (y as f64)).cos();
        }
    }
    //subtract DC component
    dbg!(m);
    dbg!(&density_dct);
    //    dbg!(&density);
    density_dct -= coefficients[[0, 0]];
    density_dct
}

fn eplace_dct_auv(density: &Array2<f64>, m: usize, u: usize, v: usize) -> f64 {
    let scale_factor = 1_f64 / (m.pow(2) as f64); //1/m^2

    let mut coefficient: f64 = 0.0; // a_u,v
    for x in 0..m {
        for y in 0..m {
            coefficient += scale_factor
                * density[[x, y]]
                * (calc_w(u, m) * (x as f64)).cos()
                * (calc_w(v, m) * (y as f64)).cos();
        }
    }
    2. * coefficient
}

fn calc_w(index: usize, m: usize) -> f64 {
    2.0 * PI * (index as f64) / (m as f64)
}
