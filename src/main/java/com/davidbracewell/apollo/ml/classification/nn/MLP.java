package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.DenseFloatMatrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.optimization.activation.Activation;
import lombok.val;

/**
 * @author David B. Bracewell
 */
public class MLP extends Classifier {
   DenseFloatMatrix w1;
   DenseFloatMatrix w2;
   Activation l1Activation = Activation.SIGMOID;
   DenseFloatMatrix b1;
   DenseFloatMatrix b2;
   Activation l2Activation = Activation.SOFTMAX;

   protected MLP(ClassifierLearner learner) {
      super(learner);
   }


   @Override
   public Classification classify(Vector vector) {
      DenseFloatMatrix in = new DenseFloatMatrix(vector.dimension(), 1, vector.toFloatArray());
      val a1 = l1Activation.apply(w1.mmul(in).addiColumnVector(b1));
      val a2 = l2Activation.apply(w2.mmul(a1).addiColumnVector(b2));
      return createResult(a2.toDoubleArray());
   }

}// END OF MLP
