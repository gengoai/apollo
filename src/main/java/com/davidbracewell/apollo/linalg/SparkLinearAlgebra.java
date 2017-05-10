package com.davidbracewell.apollo.linalg;

import com.davidbracewell.stream.StreamingContext;
import lombok.NonNull;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

/**
 * @author David B. Bracewell
 */
public final class SparkLinearAlgebra {

    private SparkLinearAlgebra() {
        throw new IllegalAccessError();
    }

    public static org.apache.spark.mllib.linalg.SingularValueDecomposition<RowMatrix, org.apache.spark.mllib.linalg.Matrix> applySVD(@NonNull RowMatrix mat, int dimension, double rCond, double tolerance) {
        return mat.computeSVD(
            dimension,
            true,
            rCond,
            Math.max(300, dimension * 3),
            tolerance,
            "auto");
    }

    public static Matrix diag(@NonNull org.apache.spark.mllib.linalg.Vector v) {
        return new DenseVector(v.toArray()).toDiagMatrix();
    }

    public static Matrix[] svd(@NonNull RowMatrix mat, int dimension, double rCond, double tolerance) {
        org.apache.spark.mllib.linalg.SingularValueDecomposition<RowMatrix, org.apache.spark.mllib.linalg.Matrix> svd = mat.computeSVD(
            dimension,
            true,
            rCond,
            Math.max(300, dimension * 3),
            tolerance,
            "auto");


        return new Matrix[]{
            toMatrix(svd.U()),
            diag(svd.s()),
            toMatrix(svd.V()),
        };
    }

    public static Matrix toMatrix(@NonNull RowMatrix m) {
        final Matrix mprime = new DenseMatrix((int) m.numRows(), (int) m.numCols());
        m
            .rows()
            .toJavaRDD()
            .zipWithIndex()
            .toLocalIterator()
            .forEachRemaining(t -> {
                mprime.setRow(t
                                  ._2()
                                  .intValue(), new DenseVector(t
                                                                   ._1()
                                                                   .toArray()));
            });
        return mprime;
    }

    public static Matrix toMatrix(@NonNull org.apache.spark.mllib.linalg.Matrix m) {
        Matrix mprime = new DenseMatrix(m.numRows(), m.numCols());
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) {
                mprime.set(i, j, m.apply(i, j));
            }
        }
        return mprime;
    }

    public static RowMatrix toRowMatrix(Matrix m) {
        JavaRDD<Vector> rdd = StreamingContext
                                  .distributed()
                                  .range(0, m.numberOfRows())
                                  .map(r ->
                                           Vectors.dense(m
                                                             .row(r)
                                                             .toArray())
                                  )
                                  .cache()
                                  .getRDD();

        return new RowMatrix(rdd.rdd());
    }


}// END OF SparkLinearAlgebra
