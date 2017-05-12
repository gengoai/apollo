package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class RandomizedSVD implements Decomposition {
    @Getter
    @Setter
    private int K = 100;
    @Getter
    @Setter
    private int numOversamples = 10;
    @Getter
    @Setter
    private int iterations = 5;


    @Override
    public Matrix[] decompose(@NonNull Matrix input) {
        LUDecomposition lu = new LUDecomposition();
        LUDecomposition qr = new LUDecomposition();
        boolean transposed = input.numberOfRows() > input.numberOfColumns();

        Matrix[] rsvd = new Matrix[]{
            new DenseMatrix(input.numberOfRows(), K),
            new DenseMatrix(K, K),
            new DenseMatrix(input.numberOfColumns(), K)
        };

        if (transposed) {
            input = input.transpose();
        }

        Matrix C = input
                       .multiply(input.transpose())
                       .toDense();
        Matrix Q = DenseMatrix.random(input.numberOfRows(), K + numOversamples);

        for (int i = 0; i < iterations; i++) {
            Q = C.multiply(Q);
            Q = lu.decompose(Q)[0];
        }

        Q = C.multiply(Q);
        Q = qr.decompose(Q)[0];

        SVD svd = new SVD(false);
        Matrix[] svdResult = svd.decompose(Q
                                               .transpose()
                                               .multiply(input));
        Matrix w = Q.multiply(svdResult[0]);


        if (transposed) {

            for (int i = 0; i < K; i++) {
                rsvd[0].setColumn(i, svdResult[2].column(i));
                rsvd[1].set(i, i, svdResult[1].get(i, i));
                rsvd[2].setColumn(i, w.column(i));
            }

        } else {

            for (int i = 0; i < K; i++) {
                rsvd[0].setColumn(i, w.column(i));
                rsvd[1].set(i, i, svdResult[1].get(i, i));
                rsvd[2].setColumn(i, svdResult[2].column(i));
            }

        }


        return rsvd;
    }

}// END OF RandomizedSVD
