use ndarray::{Array,Array2, arr2, ShapeBuilder};
use ndrustfft::{nddct2, nddct3, DctHandler};
use rustdct::DctPlanner;
use std::f64::consts::PI;

///calculate the a_u_vs from eq ( ) using an fft library
pub fn calc_coeffs(pos_density: &Array2<f64>, m: usize) -> Array2<f64> {
    
    let dc_component  = pos_density.sum(); 
    let dc_array      = Array::from_elem( (m,m), dc_component); 

    
    let density = pos_density - &dc_array; 
     
    let handler: DctHandler<f64> = DctHandler::new(m);

    let mut first_pass = Array2::<f64>::zeros((m, m));
    let mut coeffs = Array2::<f64>::zeros((m, m));

    let scale_factor = 1. / (2. * (m as f64).powi(2));

    let scaled_den = density.map(|x| x * (scale_factor));

    nddct2(&scaled_den, &mut first_pass, &handler, 1);
    nddct2(&first_pass, &mut coeffs, &handler, 0);

    
    coeffs
}


pub fn new_coeffs(density: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut coeffs = Array2::<f64>::zeros((m, m));

    let mut planner = DctPlanner::new();
    let dct2 = planner.plan_dct2(m);
     
    
    for row in 0..m {
        let mut buffer = density.row(row).to_vec();
        dct2.process_dct2(&mut buffer);
        for col in 0..m {
            coeffs[[row, col]] = buffer[col];
        }
    }

    for col in 0..m {
        let mut buffer = coeffs.column(col).to_vec();
        dct2.process_dct2(&mut buffer);
        for row in 0..m {
            coeffs[[row, col]] = buffer[row];
        }
    }
    coeffs.mapv_inplace(|x| x / (m as f64).powi(2)); //divide by m^2
    let dc_comp = coeffs[[0,0]]; 
    dbg!(&dc_comp); 
    println!(" hello yes we're drawing atttention to \n the thing above us \n pplease notice it \n is it there"); 
    coeffs.mapv_inplace(|x| x - dc_comp); 


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
       println!("post_cos");
      println!("{:.4}", &post_cos);

    //sin on each column
    for col in 0..m {
        let mut buffer2 = post_cos.column(col).to_vec();
        dst.process_dst3(&mut buffer2);
        if col ==0 {   dbg!(col, &buffer2);};
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

pub fn check_density(coeffs: &Array2<f64>, density: &Array2<f64>, m: usize) {
    let handler: DctHandler<f64> = DctHandler::new(m);
    let mut first_pass = Array2::<f64>::zeros((m, m));

    nddct3(&coeffs, &mut first_pass, &handler, 0);

    let mut density_dct = Array2::<f64>::zeros((m, m));

    nddct3(&first_pass, &mut density_dct, &handler, 1);
    density_dct.mapv_inplace(|x| x); //scale, the dct gets you something off by 2

    let test_density = &density.row(0) - &density_dct.row(0);
    let test_density_div = &density.row(0) / &density_dct.row(0);

    println!("------check density stuff");
    println!("test_density_sub, then test_density_div");
    println!("{:.4}", &test_density);
    println!("{:.4}", &test_density_div);
    println!("final retrieved density");
    println!("{:.4}", &density_dct);
    println!("--------------");
}

/// calculate the electric field in the x direction using the slow (DSCT) algorithm in the paper
pub fn eplace_elec_field_x(coeffs: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut elec_field_x = Array2::<f64>::zeros((m, m));
    for x in 0..m {
        for y in 0..m {
            elec_field_x[[x, y]] = calc_elec_point(coeffs, m, x, y);
        }
    }
    elec_field_x
}

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

///calculate the a_u_vs from eq ( ) using the slower cosine transform as directly described in the
///paper
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
    //subtract DC componentt
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
