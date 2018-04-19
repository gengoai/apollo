package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.optimization.LinearModelParameters;
import com.gengoai.apollo.ml.optimization.activation.Activation;
import com.gengoai.apollo.ml.optimization.LinearModelParameters;
import com.gengoai.apollo.ml.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class LinearModel extends Classifier implements LinearModelParameters {
   private static final long serialVersionUID = 1L;
   public NDArray weights; //num Classes x num Features
   public NDArray bias; //numClasses x 1
   public Activation activation;
   private final boolean isBinary;

   public LinearModel(ClassifierLearner learner) {
      super(learner);
      this.isBinary = learner.getEncoderPair().getLabelEncoder().size() <= 2;
   }

   public LinearModel(ClassifierLearner learner, boolean isBinary) {
      super(learner);
      this.isBinary = isBinary;
   }

   protected LinearModel(LinearModel model) {
      super(model.getPreprocessors(), model.getEncoderPair());
      this.weights = model.weights.copy();
      this.bias = model.bias.copy();
      this.activation = model.activation;
      this.isBinary = model.isBinary;
   }

   @Override
   public Classification classify(NDArray vector) {
      //vector is numFeatures x 1
      if (isBinary()) {
         double[] dist = new double[2];
         dist[1] = activation.apply(weights.dot(vector) + bias.scalarValue());
         if (activation.isProbabilistic()) {
            dist[0] = 1d - dist[1];
         } else {
            dist[0] = -dist[1];
         }
         return createResult(dist);
      }
      return createResult(activation.apply(weights.mmul(vector).addi(bias)).toArray());
   }

   @Override
   public boolean isBinary() {
      return isBinary;
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
