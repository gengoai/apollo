package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.Getter;

import java.io.Serializable;
import java.util.stream.IntStream;

/**
 * @author David B. Bracewell
 */
public class WeightMatrix implements Serializable {
   private WeightVector[] rows;
   @Getter
   private int numRows;
   @Getter
   private int numCols;

   public WeightMatrix(int nR, int nC) {
      this.rows = new WeightVector[nR];
      for (int i = 0; i < nR; i++) {
         this.rows[i] = new WeightVector(nC);
      }
      this.numRows = nR;
      this.numCols = nC;
   }

   public WeightMatrix(WeightVector weightVector) {
      this.rows = new WeightVector[]{weightVector};
      this.numRows = 1;
      this.numCols = weightVector.getWeights().dimension();
   }

   public Vector biases() {
      Vector v = Vector.sZeros(numRows);
      for (int r = 0; r < numRows; r++) {
         v.set(r, rows[r].getBias());
      }
      return v;
   }

   public Vector dot(Vector input, Activation activation) {
      Vector out = Vector.dZeros(numRows);
      IntStream.range(0, numRows)
               .parallel()
               .forEach(i -> {
                  out.set(i, activation.apply(rows[i].dot(input)));
               });
      return out;
   }

   public Vector dot(Vector input) {
      return dot(input, Activation.LINEAR);
   }

   public WeightVector get(int r) {
      return rows[r];
   }

   public Vector multiply(Vector v) {
      Preconditions.checkArgument(v.dimension() == numRows, "Dimension mismatch");
      Vector out = Vector.dZeros(numCols);
      IntStream.range(0, numCols)
               .parallel()
               .forEach(c -> {
                  v.nonZeroIterator()
                   .forEachRemaining(entry -> {
                      out.set(c, entry.getValue() * rows[entry.index].getWeights().get(c));
                   });
               });
      return out;
   }

   public void subtract(GradientMatrix gradients) {
      IntStream.range(0, numRows)
               .parallel()
               .forEach(i -> rows[i].update(gradients.gradients[i]));
   }

   public Matrix toMatrix() {
      Matrix m = SparseMatrix.zeroes(numRows, numRows);
      for (int r = 0; r < numRows; r++) {
         m.setRow(r, rows[r].getWeights());
      }
      return m;
   }

   public void update(GradientMatrix gradient) {
      for (int r = 0; r < rows.length; r++) {
         rows[r].update(gradient.gradients[r]);
      }
   }


}// END OF WeightMatrix
