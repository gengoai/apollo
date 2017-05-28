package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
@Data
public class Weights implements Serializable {
   @Getter
   @Setter
   private Matrix theta;
   @Getter
   @Setter
   private Vector bias;
   @Getter
   @Setter
   private boolean binary;

   public Weights(Matrix theta, Vector bias, boolean binary) {
      this.theta = theta;
      this.bias = bias;
      this.binary = binary;
   }

   public static Weights randomBinary(int numFeatures, double min, double max) {
      return new Weights(new SparseMatrix(SparseVector.random(numFeatures, min, max)), SparseVector.zeros(1), true);
   }

   public static Weights randomMultiClass(int numClasses, int numFeatures, double min, double max) {
      List<Vector> w = new ArrayList<>();
      for (int i = 0; i < numClasses; i++) {
         w.add(SparseVector.random(numFeatures, min, max));
      }
      return new Weights(new SparseMatrix(w), SparseVector.zeros(numClasses), false);
   }

   public Vector dot(Vector v) {
      return theta.dot(v).addSelf(bias);
   }

   public int numClasses() {
      return isBinary() ? 1 : bias.dimension();
   }

}// END OF Weights
