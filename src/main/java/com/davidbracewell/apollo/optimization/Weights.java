package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * The type Weights.
 *
 * @author David B. Bracewell
 */
@Data
@AllArgsConstructor
public class Weights implements Serializable {
   private Matrix theta;
   private Vector bias;
   private boolean binary;

   /**
    * Random binary weights.
    *
    * @param numFeatures the num features
    * @param min         the min
    * @param max         the max
    * @return the weights
    */
   public static Weights binary(int numFeatures, double min, double max) {
      return new Weights(new SparseMatrix(SparseVector.random(numFeatures, min, max)), SparseVector.zeros(1), true);
   }

   public static Weights from(@NonNull Matrix theta, @NonNull Vector bias) {
      return new Weights(theta, bias, bias.size() == 1);
   }

   /**
    * Random multi class weights.
    *
    * @param numClasses  the num classes
    * @param numFeatures the num features
    * @return the weights
    */
   public static Weights multiClass(int numClasses, int numFeatures) {
      List<Vector> w = new ArrayList<>();
      for (int i = 0; i < numClasses; i++) {
         w.add(SparseVector.zeros(numFeatures));
      }
      return new Weights(new SparseMatrix(w), SparseVector.zeros(numClasses), false);
   }

   /**
    * Random binary weights.
    *
    * @param numFeatures the num features
    * @return the weights
    */
   public static Weights randomBinary(int numFeatures) {
      return new Weights(new SparseMatrix(SparseVector.zeros(numFeatures)), SparseVector.zeros(1), true);
   }

   /**
    * Random multi class weights.
    *
    * @param numClasses  the num classes
    * @param numFeatures the num features
    * @param min         the min
    * @param max         the max
    * @return the weights
    */
   public static Weights randomMultiClass(int numClasses, int numFeatures, double min, double max) {
      List<Vector> w = new ArrayList<>();
      for (int i = 0; i < numClasses; i++) {
         w.add(SparseVector.random(numFeatures, min, max));
      }
      return new Weights(new SparseMatrix(w), SparseVector.zeros(numClasses), false);
   }

   /**
    * Dot vector.
    *
    * @param v the v
    * @return the vector
    */
   public Vector dot(Vector v) {
      return theta.dot(v).addSelf(bias);
   }

   /**
    * Num classes int.
    *
    * @return the int
    */
   public int numClasses() {
      return isBinary() ? 1 : bias.dimension();
   }

}// END OF Weights
