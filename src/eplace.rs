use crate::dct;
use crate::density;
use crate::util::B;
use crate::util::BIN_W;
use crate::util::K;
use crate::util::TARGET_DENSITY;
use crate::wirelength;
use crate::wl_grad;
use ndarray::{Array1, Array2, Axis};

pub struct NLparams {
    pub placement: Array2<f64>,
    pub ref_placement: Array2<f64>,
    pub a: f64,
    pub alpha: f64,
    pub f_k: f64,
    pub grad_f_k: Array1<f64>,
}

//get a proper name for this. the various intermediate calculations, electric fields, wirelength gradient
//things like that
pub struct CellElectricFields {
    //electric field in x direction
    x_fields: Array1<f64>,
    y_fields: Array1<f64>,
}

pub fn eplace(prev: NLparams, m: usize) -> NLparams {
    let placement = &prev.ref_placement - prev.alpha * &prev.grad_f_k;
    let a: f64 = (1. + (4. * prev.a * prev.a + 1.).sqrt()) / 2.;
    let density_overflow = calc_density_overflow(&prev.ref_placement, m);
    let gamma = calc_gamma(density_overflow);
    let wl_gradient = wl_grad::calc_wl_grad(&prev.ref_placement, gamma);

    //electric fields of the cell
    let fields = calc_cell_fields(&prev.ref_placement, m);

    let lambda: f64 = calc_lambda(&prev.ref_placement, &fields, gamma);

    let grad_f_k = calc_grad_f_k(&wl_gradient, lambda, fields);
    let ref_placement = &placement + (a - 1.) * (&placement - &prev.placement) / a;

    let density_overflow = calc_density_overflow(&placement, m);
    NLparams {
        placement: placement.clone(),
        a,
        alpha: inverse_lipschitz_constant(
            &ref_placement,
            &prev.placement,
            &grad_f_k,
            prev.grad_f_k,
        ),
        f_k: calc_f_k(&placement, density_overflow, lambda, m),
        grad_f_k,
        ref_placement,
    }
    //initially, alpha_0 is given to us as a flat 0.44 * BIN_W
    //u_k is our placement, and v_k is what's known as a "referenc placement". As we dig
    //into the algorithm and properly implement it we'll have to keep our ducks in a row
    // on it.

    //alpha * grad_f_k is scalar multiplication of an array
    //line 3 of our nesterov solver - the beating heart of the algorithm!
    //very important line of code

    //a_0 = 1

    //line 4 of nesterov solver on page 18, a is an optimization parameter, so I'm not entirely sure why this
    //is the way it is. we start with the section in the square root because rust math is a bit confusing for em
    //and here's a

    //let elec_field_x =
    //new.grad_f_k =

    //line 5 of nesterov solver , calclutating v_kplus1

    //line 6 of nesterov solver. ultimately we may want to pass in and return a struct with all the parameters and stuff
    // for each iteration of eplace
}

pub fn calc_initial_params(cell_centers: &Array2<f64>, m: usize) -> NLparams {
    let initial_density_overflow = 1.0;
    let gamma_0: f64 = calc_gamma(initial_density_overflow);
    let wl_gradient_0 = wl_grad::calc_wl_grad(cell_centers, gamma_0);
    let fields = calc_cell_fields(cell_centers, m);
    let lambda_0 = calc_lambda(cell_centers, &fields, gamma_0);

    NLparams {
        placement: cell_centers.clone(),
        ref_placement: cell_centers.clone(),
        a: 1.,
        alpha: 0.044 * 1.,
        f_k: calc_f_k(cell_centers, initial_density_overflow, lambda_0, m),
        grad_f_k: calc_grad_f_k(&wl_gradient_0, lambda_0, fields),
    }
}
pub fn calc_cell_fields(placement: &Array2<f64>, m: usize) -> CellElectricFields {
    let density = density::calc_density(placement, m); // mxm density calculation, overlaps of

    let coeffs = dct::calc_coeffs(&density, m);

    let elec_field_x = dct::elec_field_x(&coeffs, m);
    let elec_field_y = dct::elec_field_y(&coeffs, m);

    let cell_fields_x =
        placement.map_axis(Axis(1), |x| dct::apply_bins_to_cell(&x, &elec_field_x, m));
    let cell_fields_y =
        placement.map_axis(Axis(1), |x| dct::apply_bins_to_cell(&x, &elec_field_y, m));

    CellElectricFields {
        x_fields: cell_fields_x,
        y_fields: cell_fields_y,
    }
}
//eq 29

///the preconditioner depends on the detaills of vertex degree and net subsets incident to an object i
/// as well as vertex degrees. could a cell be connected to multiple nets?
/// a vertex is incident to a net if the vertex is one of the endpoints of that edge
/// the degree of a vertex is how many edges are incident to that vertex.
pub fn precondition(grad_f_k: Array1<f64>, lambda: f64) -> Array1<f64> {
    // interpretation one:  |E_i| is the number of nets a cell is connected to
    // interpretation two: |E_i| is the number of vertices in the single net that the cell's a part of

    //we will assume that |E_i| is 1 for every cell
    // lambda = lambda_0
    // q_i = 2.25

    let precon: f64 = 2. + lambda * 2.25;

    precon * grad_f_k // hopefully, f64 * Array1<f64> is implemented, we work around if not
}

fn inverse_lipschitz_constant(
    placement: &Array2<f64>,
    prev_placement: &Array2<f64>,
    grad_f_k: &Array1<f64>,
    prev_grad_f_k: Array1<f64>,
) -> f64 {
    // || placement - previous_placement||

    // || (3,5) || = (3.pow(2) + 5.pow(2)).sqrt();
    // sqrt( 3^2 + 5^2 )

    let numerator: f64 = (placement - prev_placement).map(|x| x * x).sum().sqrt();
    let denominator: f64 = (grad_f_k - prev_grad_f_k).map(|x| x * x).sum().sqrt();

    numerator / denominator
}

//equation 37, page 22
fn calc_density_overflow(placement: &Array2<f64>, m: usize) -> f64 {
    let grid_area = 1.;
    let bin_density = density::calc_density(placement, m);
    let numerator = bin_density
        .map(|bin_density| (bin_density - TARGET_DENSITY).max(0.) * grid_area)
        .sum();

    //denominator is the area of each cell summed up,
    let denominator = 2.25 * (placement.len_of(Axis(0)) as f64);

    dbg!(placement.len_of(Axis(0)));
    let tau = numerator / denominator;
    dbg!(tau);
    tau
}
pub fn calc_f_k(placement: &Array2<f64>, density_overflow: f64, lambda: f64, m: usize) -> f64 {
    let wl: f64 = wirelength::wl(placement, calc_gamma(density_overflow));

    let density = density::calc_density(placement, m);
    let coeffs = dct::calc_coeffs(&density, m);

    let total_pot: f64 = dct::total_potential(&coeffs, placement, m);
    wl + lambda * total_pot
}

//f should have the electric fields for each cell in the x and y direction
pub fn calc_lambda(placement: &Array2<f64>, fields: &CellElectricFields, gamma: f64) -> f64 {
    let wl_gradient = wl_grad::calc_wl_grad(placement, gamma);
    let numerator = wl_gradient.map(|x| x.abs()).sum();
    let denominator_x = fields.x_fields.map(|x| 2.25 * x.abs()).sum();
    let denominator_y = fields.y_fields.map(|y| 2.25 * y.abs()).sum();

    numerator / (denominator_x + denominator_y)
}

pub fn calc_gamma(density_overflow: f64) -> f64 {
    8. * BIN_W * (10_f64).powf(K * density_overflow + B)
}

pub fn calc_grad_f_k(
    wl_gradient: &Array1<f64>,
    lambda: f64,
    fields: CellElectricFields,
) -> Array1<f64> {
    precondition(wl_gradient + lambda * calc_penalty_grad(fields), lambda)
}

pub fn calc_penalty_grad(fields: CellElectricFields) -> Array1<f64> {
    let mut grad_penalty_k = Array1::<f64>::zeros(2 * fields.x_fields.len());

    for i in 0..fields.x_fields.len() {
        grad_penalty_k[i] = 2.25 * fields.x_fields[i];
        grad_penalty_k[i + 1] = 2.25 * fields.y_fields[i];
    }
    grad_penalty_k
}
