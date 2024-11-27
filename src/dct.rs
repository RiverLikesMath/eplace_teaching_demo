use crate::util::calc_w;
use ndarray::{Array, Array2, ArrayBase, Ix1, ViewRepr};
use ndrustfft::{nddct2, nddct3, DctHandler, Normalization};
use rustdct::DctPlanner;

enum Direction {
    X,
    Y,
}

enum SorC {
    Sin,
    Cos,
}

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

fn potential_coeff(w_u: f64, w_v: f64) -> f64 {
    if w_u == 0. && w_v == 0. {
        0.
    } else {
        1. / (w_u.powi(2) + w_v.powi(2))
    }
}

fn elec_coeff(u: usize, v: usize, m: usize, dir: Direction) -> f64 {
    let w_u = calc_w(u, m);
    let w_v = calc_w(v, m);

    let mut elec_coeff = 0.;

    if u != 0 && v != 0 {
        match dir {
            Direction::X => elec_coeff = w_u * potential_coeff(w_u, w_v),
            Direction::Y => elec_coeff = w_v * potential_coeff(w_u, w_v),
        }
    }
    elec_coeff
}

fn fft_row_or_col(
    row_col: &mut ArrayBase<ViewRepr<&mut f64>, Ix1>,
    planner: &mut DctPlanner<f64>,
    transform: SorC,
    m: usize,
) {
    let fft;

    match transform {
        SorC::Sin => fft = planner.plan_dst3(m),
        SorC::Cos => fft = planner.plan_dct3(m),
    }

    let mut buffer = row_col.to_vec();

    match transform {
        SorC::Sin => fft.process_dst3(&mut buffer),
        SorC::Cos => fft.process_dct3(&mut buffer),
    }

    let temp = Array::from_vec(buffer);
    row_col.assign(&temp);
}

pub fn elec_field_x(coeffs: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut elec_x = Array2::<f64>::zeros((m, m));

    let mut planner = DctPlanner::new();

    for u in 0..m {
        for v in 0..m {
            elec_x[[u, v]] = coeffs[[u, v]] * elec_coeff(u, v, m, Direction::X);
        }
    }

    //inverse cos transform on each row
    for mut row in elec_x.rows_mut() {
        fft_row_or_col(&mut row, &mut planner, SorC::Cos, m);
    }

    // inverse sin transform on each column
    for mut col in elec_x.columns_mut() {
        fft_row_or_col(&mut col, &mut planner, SorC::Sin, m);
    }

    elec_x
}

///this code is similar to elec_field_x, but writing it out to make it clear. equation 24, second half)

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

/*
keeping this just as an example of how to use rustdct
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
*/
