#![allow(dead_code)]
use crate::dct::elec_field_x;
use ndarray::Array2;
///eventually these should all be shifted to rust's built in testing functionality
use ndrustfft::{nddct3, DctHandler};

pub fn check_density(coeffs: &Array2<f64>, density: &Array2<f64>, m: usize) {
    let handler: DctHandler<f64> = DctHandler::new(m); //.normalization(Normalization::None);
    let mut first_pass = Array2::<f64>::zeros((m, m));
    let mut density_dct = Array2::<f64>::zeros((m, m));

    nddct3(&coeffs, &mut first_pass, &handler, 0);
    nddct3(&first_pass, &mut density_dct, &handler, 1);

    let test_density = &density.row(0) - &density_dct.row(0);
    let test_density_div = &density.row(0) / &density_dct.row(0);

    println!("------check density stuff");
    println!("test density diff then div");
    println!("{:.4}", &test_density);
    println!("{:.4}", &test_density_div);
}

pub fn test_elec_field_x(coeffs: &Array2<f64>, good: &Array2<f64>, m: usize) {
    let fast_elec_x = elec_field_x(&coeffs, m);

    let row = 7;

    let diff = &good.row(row) - &fast_elec_x.row(row);
    let div = &good.row(row) / &fast_elec_x.row(row);

    println!("x electric field, difference ref from fft");
    println!("{:.4}", diff);
    println!("x electr field, ration of ref and fft");
    println!("{:.4}", div);
}
