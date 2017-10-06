package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.optimization.LinearModelParameters;
import com.davidbracewell.apollo.ml.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class LinearModel extends Classifier implements LinearModelParameters {
   private static final long serialVersionUID = 1L;
   public NDArray weights; //num Classes x num Features
   public NDArray bias; //numClasses x 1
   public Activation activation;

   public LinearModel(ClassifierLearner learner) {
      super(learner);
   }

   @Override
   public Classification classify(NDArray vector) {
      //vector is numFeatures x 1
      return createResult(activation.apply(weights.mmul(vector).addi(bias)).toArray());
   }

   @Override
   public Activation getActivation() {
      return activation;
   }

   @Override
   public NDArray getBias() {
      return bias;
   }

   @Override
   public NDArray getWeights() {
      return weights;
   }

   @Override
   public int numberOfLabels() {
      return getEncoderPair().numberOfLabels();
   }

}// END OF GLM
