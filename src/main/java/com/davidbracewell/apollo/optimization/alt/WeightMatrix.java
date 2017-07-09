package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.base.Stopwatch;

import java.io.Serializable;
import java.util.stream.IntStream;

/**
 * @author David B. Bracewell
 */
public class WeightMatrix implements Serializable {
   private WeightVector[] rows;
   private int numRows;
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

   public static void main(String[] args) {
      WeightMatrix layer1 = new WeightMatrix(256, 784);
      WeightMatrix layer2 = new WeightMatrix(9, 256);

      Stopwatch sw = Stopwatch.createStarted();
      for (int i = 0; i < 100_000; i++) {
         Vector a0 = SparseVector.random(784, -1, 1);
         Vector a1 = layer1.dot(a0, Activation.SIGMOID);
         Vector a2 = layer2.dot(a1, Activation.SIGMOID);
         Vector a3Delta = a2.subtract(Vector.sZeros(9).set((int) (Math.random() * 9), 1));
         Vector a2Delta = layer2.multiply(a3Delta);
         GradientMatrix a2Gradient = new GradientMatrix(a1, a3Delta);
         GradientMatrix a1Gradient = new GradientMatrix(a0, a2Delta.multiply(Activation.SIGMOID.valueGradient(a1)));
         layer1.subtract(a1Gradient);
         layer2.subtract(a2Gradient);
         if (i % 100 == 0) {
            System.out.println(i + ": " + sw);
         }
      }
      sw.stop();
      System.out.println(sw);

   }

   public Vector dot(Vector input) {
      return dot(input, Activation.LINEAR);
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


}// END OF WeightMatrix
