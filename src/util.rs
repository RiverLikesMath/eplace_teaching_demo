use std::f64::consts::PI;

///calculates the w_u and w_v used in equations 21-25 or so
pub fn calc_w(index: usize, m: usize) -> f64 {
    2.0 * PI * (index as f64) / (m as f64)
}
