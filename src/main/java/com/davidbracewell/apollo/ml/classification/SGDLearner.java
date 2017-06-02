package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxFunction;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.regularization.L1Regularization;
import com.davidbracewell.apollo.optimization.regularization.WeightUpdater;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class SGDLearner extends ClassifierLearner {
   @Getter
   @Setter
   private LearningRate learningRate = new ConstantLearningRate(0.1);
   @Getter
   @Setter
   private WeightUpdater weightUpdater = new L1Regularization(0.1);
   @Getter
   @Setter
   private LossFunction loss = new LogLoss();
   @Getter
   @Setter
   private Activation activation = new SoftmaxFunction();
   @Getter
   @Setter
   private int maxIterations = 300;
   @Getter
   @Setter
   private double tolerance = 1e-9;
   @Getter
   @Setter
   private boolean verbose = false;

   @Override
   public void reset() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      Optimizer sgd = new BatchOptimizer(new SGD(), 20);

      Weights start;
      if (dataset.getLabelEncoder().size() <= 2) {
         start = Weights.binary(dataset.getFeatureEncoder().size());
      } else {
         start = Weights.multiClass(dataset.getLabelEncoder().size(), dataset.getFeatureEncoder().size());
      }


      Weights weights = sgd.optimize(start,
                                     dataset::asFeatureVectors,
                                     new LogisticCostFunction(),
                                     TerminationCriteria.create().maxIterations(maxIterations).historySize(3).tolerance(tolerance),
                                     learningRate,
                                     weightUpdater,
                                     verbose).getWeights();

      GLM glm = new GLM(dataset.getEncoderPair(), dataset.getPreprocessors());
      glm.weights = weights;
      glm.activation = activation;

      return glm;
   }
}// END OF SGDLearner
