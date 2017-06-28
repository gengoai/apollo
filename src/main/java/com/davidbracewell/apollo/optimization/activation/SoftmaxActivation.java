package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class SoftmaxActivation implements DifferentiableActivation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return SigmoidActivation.INSTANCE.apply(x);
   }

   @Override
   public Vector apply(Vector x) {
      double max = x.max();
      x.mapSelf(d -> Math.exp(d - max));
      double sum = x.sum();
      return x.mapDivideSelf(sum);
   }

   @Override
   public boolean isProbabilistic() {
      return true;
   }

   @Override
   public Matrix valueGradient(@NonNull Matrix predicted, @NonNull Matrix actual) {
      Matrix gradient = new DenseMatrix(predicted.numberOfRows(), predicted.numberOfColumns());
      for (int columnIndex = 0; columnIndex < predicted.numberOfColumns(); columnIndex++) {
         Vector predictedColumn = predicted.column(columnIndex);
         Vector actualColumn = actual.column(columnIndex);
         for (int outerRow = 0; outerRow < predicted.numberOfRows(); outerRow++) {
            double derivative = 0;
            double predictedValue = predictedColumn.get(outerRow);

            for (int innerRow = 0; innerRow < predicted.numberOfRows(); innerRow++) {
               double actualValue = actualColumn.get(innerRow);
               if (outerRow == innerRow) {
                  derivative += actualValue * predictedValue * (1.0 - predictedValue);
               } else {
                  derivative += actualValue * (-predictedValue * predictedValue);
               }
            }
            gradient.set(outerRow, columnIndex, derivative);
         }
      }
      return gradient;
   }

   @Override
   public Vector valueGradient(Vector activated) {
      return activated;
   }

   @Override
   public Vector valueGradient(@NonNull Vector predicted, @NonNull Vector actual) {
      Vector gradient = new DenseVector(predicted.dimension());
      for (int i = 0; i < predicted.dimension(); i++) {
         double derivative = 0;
         double predictedValue = predicted.get(i);
         for (int j = 0; j < actual.dimension(); j++) {
            double actualValue = actual.get(j);
            if (i == j) {
               derivative += actualValue * predictedValue * (1.0 - predictedValue);
            } else {
               derivative += actualValue * (-predictedValue * predictedValue);
            }
         }
         gradient.set(i, derivative);
      }
      return gradient;
   }

}// END OF SoftmaxActivation
