package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import lombok.val;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * @author David B. Bracewell
 */
public class MLP extends Classifier {
   DoubleMatrix w1;
   DoubleMatrix w2;
   DoubleMatrix b1;
   DoubleMatrix b2;

   protected MLP(ClassifierLearner learner) {
      super(learner);
   }

   public static DoubleMatrix dsigmoid(DoubleMatrix in) {
      return in.mul(DoubleMatrix.ones(in.rows, in.columns).subi(in));
   }

   public static DoubleMatrix relu(DoubleMatrix in){
      return in.max(0);//DoubleMatrix.zeros(in.length));
   }

   public static DoubleMatrix drelu(DoubleMatrix in) {
      return in.gt(0);
   }


   public static DoubleMatrix sigmoid(DoubleMatrix in) {
      return MatrixFunctions.expi(in.neg())
                            .addi(1.0f).rdiv(1.0f);
   }

   public static DoubleMatrix softmax(DoubleMatrix in) {
      val max = in.columnMaxs();
      val exp = MatrixFunctions.exp(in.subRowVector(max));
      val sums = exp.columnSums();
      return exp.diviRowVector(sums);
   }

   @Override
   public Classification classify(Vector vector) {
      DoubleMatrix in = new DoubleMatrix(vector.dimension(), 1, vector.toArray());
      val a1 = sigmoid((w1.mmul(in)).addiColumnVector(b1));
      val a2 = softmax((w2.mmul(a1)).addiColumnVector(b2));
      return createResult(a2.toArray());
   }

   private double[] toDouble(float[] array) {
      double[] d = new double[array.length];
      for (int i = 0; i < array.length; i++) {
         d[i] = array[i];
      }
      return d;
   }
}// END OF MLP
