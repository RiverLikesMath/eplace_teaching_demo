#![allow(dead_code)]

use crate::{
    dct::{self, elec_field_x},
    density, eplace,
    ref_dct::ref_dct_coeffs,
    wl_grad::calc_wl_grad,
};
use ndarray::{Array2, Axis};
///eventually these should all be shifted to rust's built in testing functionality
use ndrustfft::{nddct3, DctHandler};

pub fn check_density(coeffs: &Array2<f64>, density: &Array2<f64>, m: usize) {
    let handler: DctHandler<f64> = DctHandler::new(m); //.normalization(Normalization::None);
    let mut first_pass = Array2::<f64>::zeros((m, m));
    let mut density_dct = Array2::<f64>::zeros((m, m));

    nddct3(coeffs, &mut first_pass, &handler, 0);
    nddct3(&first_pass, &mut density_dct, &handler, 1);

    let test_density = &density.row(0) - &density_dct.row(0);
    let test_density_div = &density.row(0) / &density_dct.row(0);

    println!("------check density stuff");
    println!("test density diff then div");
    println!("{test_density:.4}");
    println!("{test_density_div:.4}");
}

pub fn test_elec_field_x(coeffs: &Array2<f64>, good: &Array2<f64>, m: usize) {
    let fast_elec_x = elec_field_x(coeffs, m);

    let row = 7;

    let diff = &good.row(row) - &fast_elec_x.row(row);
    let div = &good.row(row) / &fast_elec_x.row(row);

    println!("x electric field, difference ref from fft");
    println!("{diff:.4}");
    println!("x electr field, ration of ref and fft");
    println!("{div:.4}");
}

pub fn check_dc_component(cell_centers: &Array2<f64>, m: usize) {
    let density_b4_sub = density::initial_density(cell_centers, m);

    let ref_auvs = ref_dct_coeffs(&density_b4_sub, m);
    let ref_dc = ref_auvs[[0, 0]];

    let fft_auvs = dct::calc_coeffs(&density_b4_sub, m);
    let fft_dc = fft_auvs[[0, 0]];

    let dc_without_fft = density::dc_component(&density_b4_sub, m);

    println!("reference dc calculated from the reference 2d cosine transform minus the dc calculated solely from density");
    let dc = dc_without_fft[[0, 0]];
    dbg!(ref_dc - dc);

    println!("same, but using the fft dc calculation");
    dbg!(fft_dc - dc);
}

pub fn shape_test(cell_centers: &Array2<f64>, m: usize) {
    dbg!(&cell_centers);

    let fields = eplace::calc_cell_fields(cell_centers, m);

    let wl_gradient = calc_wl_grad(cell_centers, 0.8);
    let lambda = eplace::calc_lambda(cell_centers, &fields, 0.8);
    let grad = eplace::calc_grad_f_k(&wl_gradient, lambda, fields);
    dbg!(&grad);
    dbg!(grad.len_of(Axis(0)));

    //should be arranged in a similar format to cell_centers
    dbg!(grad
        .to_shape((grad.len_of(Axis(0)) / 2, 2))
        .map_err(|err| println!("{:?}", err))
        .ok());
}
