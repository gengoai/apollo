package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.base.Stopwatch;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

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
      INDArray layer1 = Nd4j.rand(256, 100_000);
      INDArray layer2 = Nd4j.rand(1, 256);

      Stopwatch sw = Stopwatch.createStarted();
      for (int i = 0; i < 25_000; i++) {
         INDArray a0 = Nd4j.rand(1, 100_000);
         INDArray a1 = sigmoid(layer1.mmul(a0.transposei()));
         INDArray a2 = sigmoid(layer2.mmul(a1));

         INDArray a3Delta = a2.sub(Nd4j.create(1));
         INDArray a2Delta = a3Delta.mmul(layer2);
         INDArray a1Delta = a2Delta.mmul(layer1);

         INDArray a2Gradient = a1.mmul(a3Delta);
         INDArray a1Gradient = a0.transposei().mmul(a2Delta).transposei();

         layer1.subi(a1Gradient);
         layer2.subi(a2Gradient);

         if (i % 1000 == 0) {
            System.out.println(i + ": " + sw.elapsed(TimeUnit.SECONDS));
         }

      }

      System.out.println(sw.elapsed(TimeUnit.SECONDS));
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
