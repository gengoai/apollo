package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.stream.StreamingContext;
import lombok.Data;
import lombok.NonNull;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;


/**
 * @author David B. Bracewell
 */
@Data
public class DistributedSVD implements Decomposition {
    private int dimension = 100;
    private double tolerance = 1e-10;
    private double rCond = 1e-9;

    private Matrix convert(org.apache.spark.mllib.linalg.Matrix m) {
        Matrix mprime = new DenseMatrix(m.numRows(), m.numCols());
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) {
                mprime.set(i, j, m.apply(i, j));
            }
        }
        return mprime;
    }

    private Matrix convert(org.apache.spark.mllib.linalg.Vector v) {
        double[] va = v.toArray();
        Matrix mprime = new SparseMatrix(va.length, va.length);
        for (int i = 0; i < va.length; i++) {
            mprime.set(i, i, va[i]);
        }
        return mprime;
    }


    private Matrix convert(RowMatrix m) {
        Matrix mprime = new DenseMatrix((int) m.numRows(), (int) m.numCols());
        m
            .rows()
            .toJavaRDD()
            .zipWithIndex()
            .foreach(t -> {
                mprime.setRow(t
                                  ._2()
                                  .intValue(), new DenseVector(t
                                                                   ._1()
                                                                   .toArray()));
            });
        return mprime;
    }

    @Override
    public Matrix[] decompose(@NonNull Matrix input) {
        JavaRDD<Vector> rdd = StreamingContext
                                  .distributed()
                                  .range(0, input.numberOfRows())
                                  .map(r ->
                                           Vectors.dense(input
                                                             .row(r)
                                                             .toArray())
                                  )
                                  .cache()
                                  .getRDD();

        RowMatrix mat = new RowMatrix(rdd.rdd());
        org.apache.spark.mllib.linalg.SingularValueDecomposition<RowMatrix, org.apache.spark.mllib.linalg.Matrix> svd = mat.computeSVD(
            dimension,
            true,
            rCond,
            Math.max(300, dimension * 3),
            tolerance,
            "auto");


        return new Matrix[]{
            convert(svd.U()),
            convert(svd.s()),
            convert(svd.V()),
        };
    }

}// END OF DistributedSVD
