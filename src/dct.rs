use crate::util::{calc_w, overlap};
use ndarray::{Array, Array2, ArrayBase, ArrayView1, Axis, Ix1, ViewRepr};
use ndrustfft::{nddct2, nddct3, DctHandler, Normalization};
use rustdct::DctPlanner;

#[derive(Clone, Copy)]
enum Direction {
    X,
    Y,
}

enum SorC {
    Sin,
    Cos,
}

///calculate the potential energy for each bin, then find the overlap of the potential
/// with each cell so we can multiply to find each cell's potential. Then we'll add up all
/// the cell potentials and divide by 2. From equation (insert eq here, pg# here). There
/// may be a faster way to do this, but this is the most direct implementation of the paper's
/// method.We'll use the a_uv coefficients calculated earlier as a parameter
pub fn total_potential(coeffs: &Array2<f64>, cell_centers: &Array2<f64>, m: usize) -> f64 {
    // multiply each a_uv by 1/ (w_u^2+w_v^2)
    let mut pot_coeffs = Array2::<f64>::zeros((m, m));

    //once again wondering if there's a fancy iterator way to do this, but my
    //maps and flatmaps are currently failing me since it's 2D.
    for u in 0..m {
        for v in 0..m {
            pot_coeffs[[u, v]] = coeffs[[u, v]] * potential_coeff(calc_w(u, m), calc_w(v, m));
        }
    }

    //potential in each bin is an inverse 2D cosine on the updated coefficients
    let pot_bins = inverse_2ddct(&pot_coeffs, m);

    //don't forget to divide by 2!
    cell_centers
        .map_axis(Axis(0), |x| apply_bins_to_cell(&x, &pot_bins, m))
        .sum()
        / 2.0
}

///given either the electric field in the x or y direction or the potential for each bin
/// multiply that value by the overlap with each cell , sum, and return
pub fn apply_bins_to_cell(cell_loc: &ArrayView1<f64>, bin_data: &Array2<f64>, m: usize) -> f64 {
    let (cell_u, cell_v) = (cell_loc[0] as usize, cell_loc[1] as usize);
    let (u_start, u_end, v_start, v_end) = bounds_check(cell_u, cell_v, m);

    //is there a way to do this with better iterators or the like, make it prettier?

    // well, if you insist.

    (u_start..u_end)
        .flat_map(|u| {
            (v_start..v_end).map(move |v| overlap(cell_loc, u as f64, v as f64) * bin_data[[u, v]])
        })
        .sum::<f64>()
}

///calculate the a_u_vs from eq ( ) using an fft library
pub fn calc_coeffs(density: &Array2<f64>, m: usize) -> Array2<f64> {
    let handler: DctHandler<f64> = DctHandler::new(m).normalization(Normalization::None);

    let mut first_pass = Array2::<f64>::zeros((m, m));
    let mut coeffs = Array2::<f64>::zeros((m, m));

    //cosine transform on the rows
    nddct2(density, &mut first_pass, &handler, 0);

    //cosine transform on the columns
    nddct2(&first_pass, &mut coeffs, &handler, 1);

    coeffs.mapv_inplace(|x| x / ((m as f64).powi(2)));

    coeffs
}

///1/(w_u^2 + w_v^2), except when w_u== w_v ==0 , the amount each a_u_v needs to be
/// multiplied by in order to be ready for the 2d inverse dct to get potential, via
/// eq () on page ()
fn potential_coeff(w_u: f64, w_v: f64) -> f64 {
    if w_u == 0. && w_v == 0. {
        return 0.;
    }

    1. / (w_u.powi(2) + w_v.powi(2))
}

///calculate the modifier to a given a_uv needed to calculate electric field in
/// a given direction
fn elec_coeff(u: usize, v: usize, m: usize, dir: Direction) -> f64 {
    let w_u = calc_w(u, m);
    let w_v = calc_w(v, m);

    if u == 0 && v == 0 {
        return 0.;
    }

    match dir {
        Direction::X => w_u * potential_coeff(w_u, w_v),
        Direction::Y => w_v * potential_coeff(w_u, w_v),
    }
}

///2d inverse DCT for the potential calculation. The code is similar to calc_coeffs, but
/// I'm leaving as is until I can figure a way to combine them
fn inverse_2ddct(buffer: &Array2<f64>, m: usize) -> Array2<f64> {
    //normalization will probably have to be updated later
    let handler: DctHandler<f64> = DctHandler::new(m).normalization(Normalization::None);

    let mut first_pass = Array2::<f64>::zeros((m, m));
    let mut inverse = Array2::<f64>::zeros((m, m));

    //cosine transform on the rows
    nddct3(buffer, &mut first_pass, &handler, 0);

    //cosine transform on the columns
    nddct3(&first_pass, &mut inverse, &handler, 1);

    inverse
}

///take an fft for the electric field calculations
fn fft_row_or_col(
    row_col: &mut ArrayBase<ViewRepr<&mut f64>, Ix1>,
    planner: &mut DctPlanner<f64>,
    transform: SorC,
    m: usize,
) {
    let mut buffer = row_col.to_vec();

    match transform {
        SorC::Sin => planner.plan_dst3(m).process_dst3(&mut buffer),
        SorC::Cos => planner.plan_dct3(m).process_dct3(&mut buffer),
    }

    row_col.assign(&Array::from_vec(buffer));
}

///    this will calculate the electric field in the x direction for each bin
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
pub fn elec_field_y(coeffs: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut planner = DctPlanner::new();
    let mut elec_y = Array2::<f64>::zeros((m, m));
    for u in 0..m {
        for v in 0..m {
            elec_y[[u, v]] = coeffs[[u, v]] * elec_coeff(u, v, m, Direction::Y);
        }
    }

    //inverse sin transform on each row
    for mut row in elec_y.rows_mut() {
        fft_row_or_col(&mut row, &mut planner, SorC::Sin, m);
    }

    // inverse cos transform on each column
    for mut col in elec_y.columns_mut() {
        fft_row_or_col(&mut col, &mut planner, SorC::Cos, m);
    }
    elec_y
}

// returns the electric field in a given direction for a cell. Expects an Array1 (may switch to tuple?)
// with the x and y coordinates of the cell. Has to take into account the overlap with all of the bins the cell
// is in. In eplace proper that'd include the stretching dsecribed in (page #), but al of our cells are larger
//than a bin so there's no stretching. Note we *could* use types to enforce that cell_loc is a 1d array of length
//2, we won't for now

///helper function for elec_field_cell, should we create an enum for it?
fn bounds_check(cell_u: usize, cell_v: usize, m: usize) -> (usize, usize, usize, usize) {
    let u_start = if cell_u == 0 { 0 } else { cell_u - 1 };
    let v_start = if cell_v == 0 { 0 } else { cell_v - 1 };
    let u_end = if cell_u == m - 1 { m } else { cell_u + 2 };
    let v_end = if cell_v == m - 1 { m } else { cell_v + 2 };
    (u_start, u_end, v_start, v_end)
}
