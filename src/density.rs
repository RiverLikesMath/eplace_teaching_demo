use ndarray::{Array, Array2, Axis};

const BIN_W: f64 = 1.;

fn add_density(density: &mut Array2<f64>, x: usize, y: usize, added_density: f64) {
    density[[x, y]] += added_density;
}

///calculates the density given the current location of each centr, by calculating overlap of each cell w/ bins
///there's probably a shorter clearer way to do this
pub fn calc_density(cell_centers: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut density = Array2::<f64>::zeros((m, m));
    for cell in 0..cell_centers.len_of(Axis(0)) {
        //calculate which bins its in
        //calculate the overlap with each bin
        //update the density array/matrix accordingly

        let left_edge = cell_centers[[cell, 0]] - 0.75;
        let right_edge = cell_centers[[cell, 0]] + 0.75;
        let upper_edge = cell_centers[[cell, 1]] + 0.75;
        let lower_edge = cell_centers[[cell, 1]] - 0.75;

        let left_edge_bin = left_edge.floor() as usize;
        let right_edge_bin = right_edge.floor() as usize;
        let upper_edge_bin = upper_edge.floor() as usize;
        let lower_edge_bin = lower_edge.floor() as usize;

        let right_width = right_edge.fract();
        let left_width = (left_edge.floor() + BIN_W) - left_edge;

        let upper_height = upper_edge.fract();
        let lower_height = (lower_edge.floor() + BIN_W) - lower_edge;

        //update the upper right corner
        add_density(
            &mut density,
            right_edge_bin,
            upper_edge_bin,
            right_width * upper_height,
        );
        //update the lower right corner
        add_density(
            &mut density,
            right_edge_bin,
            lower_edge_bin,
            right_width * lower_height,
        );
        //update the lower left corner
        add_density(
            &mut density,
            left_edge_bin,
            lower_edge_bin,
            left_width * lower_height,
        );
        //update the upper left corner
        add_density(
            &mut density,
            left_edge_bin,
            upper_edge_bin,
            left_width * upper_height,
        );

        for y in (lower_edge_bin + 1)..upper_edge_bin {
            //now for the left edges that aren't corners
            add_density(&mut density, left_edge_bin, y, left_width); //height of a bin is 1
                                                                              //right edges
            add_density(&mut density, right_edge_bin, y, right_width);
        }

        for x in (left_edge_bin + 1)..right_edge_bin {
            //now for the upper edges that aren't corners
            add_density(&mut density, x, upper_edge_bin, upper_height); //width of a bin is 1
                                                                                 //lower edges
            add_density(&mut density, x, lower_edge_bin, lower_height);
        }

        //add density of completely filled bins
        for x in (left_edge_bin + 1)..right_edge_bin {
            for y in (lower_edge_bin + 1)..upper_edge_bin {
                add_density(&mut density, x, y, 1.);
            }
        }
    }

    //subtract the DC component (the total density / m^2 )
    let dc_component = density.sum() / (m as f64).powi(2);
    let dc_array = Array::from_elem((m, m), dc_component);
    
    density - dc_array
}
