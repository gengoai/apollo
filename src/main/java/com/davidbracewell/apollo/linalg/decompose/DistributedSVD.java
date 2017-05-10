package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.Matrix;
import lombok.Data;
import lombok.NonNull;

import static com.davidbracewell.apollo.linalg.SparkLinearAlgebra.svd;
import static com.davidbracewell.apollo.linalg.SparkLinearAlgebra.toRowMatrix;


/**
 * @author David B. Bracewell
 */
@Data
public class DistributedSVD implements Decomposition {
    private int dimension = 100;
    private double tolerance = 1e-10;
    private double rCond = 1e-9;

    @Override
    public Matrix[] decompose(@NonNull Matrix input) {
        return svd(toRowMatrix(input), dimension, rCond, tolerance);
    }

}// END OF DistributedSVD
