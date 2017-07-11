package com.davidbracewell.apollo.optimization;

import com.davidbracewell.Copyable;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.alt.WeightMatrix;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The type Weights.
 *
 * @author David B. Bracewell
 */
@Data
@AllArgsConstructor
public class Weights implements Serializable, Copyable<Weights> {
   private Matrix theta;
   private Vector bias;
   private boolean binary;

   public Weights(int numberOfRows, int numberOfColumns, WeightInitializer weightInitializer) {
      this.theta = weightInitializer.initialize(SparseMatrix.zeroes(numberOfRows, numberOfColumns));
      this.bias = SparseVector.zeros(numberOfRows);
      this.binary = numberOfColumns <= 2;

   }

   public Weights(WeightMatrix matrix) {
      this.theta = matrix.toMatrix();
      this.bias = matrix.biases();
   }

   @Override
   public Weights copy() {
      return new Weights(theta.copy(), bias.copy(), binary);
   }

   /**
    * Dot vector.
    *
    * @param v the v
    * @return the vector
    */
   public Vector dot(Vector v) {
      return theta.dot(v).column(0).copy().addSelf(bias);
   }

   /**
    * Num classes int.
    *
    * @return the int
    */
   public int numClasses() {
      return isBinary() ? 1 : bias.dimension();
   }

   /**
    * Set.
    *
    * @param other the other
    */
   public void set(@NonNull Weights other) {
      this.theta = other.theta;
      this.binary = other.binary;
      this.bias = other.bias;
   }

}// END OF Weights
