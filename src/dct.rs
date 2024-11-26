use crate::util::calc_w;
use ndarray::Array2;
use ndrustfft::{nddct2, nddct3, DctHandler, Normalization};
use rustdct::DctPlanner;

///calculate the a_u_vs from eq ( ) using an fft library
pub fn calc_coeffs(density: &Array2<f64>, m: usize) -> Array2<f64> {
    let handler: DctHandler<f64> = DctHandler::new(m).normalization(Normalization::None);

    let mut first_pass = Array2::<f64>::zeros((m, m));
    let mut coeffs = Array2::<f64>::zeros((m, m));

    //cosine transform on the rows
    nddct2(&density, &mut first_pass, &handler, 0);

    //cosine transform on the columns
    nddct2(&first_pass, &mut coeffs, &handler, 1);

    coeffs.mapv_inplace(|x| x / ((m as f64).powi(2)));

    coeffs
}

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

pub fn new_coeffs(density: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut coeffs = Array2::<f64>::zeros((m, m));

    let mut planner = DctPlanner::new();
    let dct2 = planner.plan_dct2(m);

    //run a cosine transform on each row
    for row in 0..m {
        let mut buffer = density.row(row).to_vec();
        dct2.process_dct2(&mut buffer);
        for col in 0..m {
            coeffs[[row, col]] = buffer[col];
        }
    }

    //run another cosine transform on the columns that result
    for col in 0..m {
        let mut buffer = coeffs.column(col).to_vec();
        dct2.process_dct2(&mut buffer);
        for row in 0..m {
            coeffs[[row, col]] = buffer[row];
        }
    }
    coeffs.mapv_inplace(|x| x / (m as f64).powi(2)); //divide by m^2

    coeffs
}

fn potential_coeff(w_u: f64, w_v: f64) -> f64 {
    if w_u == 0. && w_v == 0. {
        0.
    } else {
        1. / (w_u.powi(2) + w_v.powi(2))
    }
}
fn elec_coeff_x(u: usize, v: usize, m: usize) -> f64 {
    let w_u = calc_w(u, m);
    let w_v = calc_w(v, m);

    let mut elec_coeff = 0.;

    if (u != 0 && v != 0) {
        elec_coeff = w_u * potential_coeff(w_u, w_v)
    }
    elec_coeff
}
pub fn elec_field_x(coeffs: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut elec_x = Array2::<f64>::zeros((m, m));

    let mut elec_coeffs = Array2::<f64>::zeros((m, m));

    for u in 0..m {
        for v in 0..m {
            elec_coeffs[[u, v]] = coeffs[[u, v]] * elec_coeff_x(u, v, m);
        }
    }
    let mut planner = DctPlanner::new();
    let dct = planner.plan_dct3(m);
    let dst = planner.plan_dst3(m);

    let mut post_cos = Array2::<f64>::zeros((m, m));

    //cos on each row
    for row in 0..m {
        let mut buffer = elec_coeffs.row(row).to_vec();
        dct.process_dct3(&mut buffer);
        for col in 0..m {
            if row == 0 {
                post_cos[[row, col]] = 0.;
            } else {
                post_cos[[row, col]] = buffer[col];
            }
        }
    }

    //sin on each column
    for col in 0..m {
        let mut buffer2 = post_cos.column(col).to_vec();
        dst.process_dst3(&mut buffer2);

        for row in 0..m {
            elec_x[[row, col]] = buffer2[row];
        }
    }
    elec_x
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
