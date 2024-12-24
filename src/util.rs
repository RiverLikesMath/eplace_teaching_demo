///functions used in multiple files
use ndarray::ArrayView1;
use std::f64::consts::PI;

pub const BIN_W: f64 = 1.;
pub const K: f64 = 20. / 9.;
pub const B: f64 = -11. / 9.;
pub const TARGET_DENSITY: f64 = 0.9;

///calculates the w_u and w_v used in equations 21-25 or so
pub fn calc_w(index: usize, m: usize) -> f64 {
    2.0 * PI * (index as f64) / (m as f64)
}

///at some point we may rewrite density to use this
pub fn overlap(cell_loc: &ArrayView1<f64>, u: f64, v: f64) -> f64 {
    let left_edge = cell_loc[0] - 0.75;
    let right_edge = cell_loc[0] + 0.75;
    let upper_edge = cell_loc[1] + 0.75;
    let lower_edge = cell_loc[1] - 0.75;

    let mut x_overlap = right_edge.min(u + 1.) - left_edge.max(u);
    let mut y_overlap = upper_edge.max(v + 1.) - lower_edge.max(v);

    if x_overlap < 0. {
        x_overlap = 0.;
    };
    if y_overlap < 0. {
        y_overlap = 0.;
    };

    x_overlap * y_overlap
}

// density needs to calculate the overlap of each cell and then add that overlap to the bin density ()
// elec_x needs to calculate the overlap of each cell, multiply it by electric field of a given direction of each bin,
// and then add that to the cell elec field in that direction

//possible rewrite of density function, but tbh, why??
/*
fn density(cell_centers: &mut Array2<f64>, arr: &Array1<f64>,  m: usize) -> f64 {

    let mut density = Array2::<f64>::zeros((m, m));

    for cell in 0..cell_centers.len_of(Axis(0)) {
        //calculate which bins its in
        //calculate the overlap with each bin
        //update the density array/matrix accordingly


        for u in cell_u -1 .. cell_u +2 {
            for v in cell_v-1 .. cell_v + 2 {
                if (0..m).contains(&u) && (0..m).contains(&v){
                    density[[u,v]]   += overlap(&cell_loc, u,v,m);
                    elec_field[cell] += overlap(&cell_loc,u,v,m) * bins_elec_field[[u,v]];
                }
            }
        }
    }
    density
}

*/
