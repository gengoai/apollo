package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.nn.SGD;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxActivation;
import com.davidbracewell.apollo.optimization.loss.CrossEntropyLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.update.DeltaRule;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class SoftmaxLearner extends ClassifierLearner {
   private final LossFunction loss = new CrossEntropyLoss();
   private final Activation activation = new SoftmaxActivation();
   @Getter
   @Setter
   private LearningRate learningRate = new ConstantLearningRate(0.1);
   @Getter
   @Setter
   private WeightUpdate weightUpdater = new DeltaRule();
   @Getter
   @Setter
   private int maxIterations = 300;
   @Getter
   @Setter
   private int batchSize = 20;
   @Getter
   @Setter
   private double tolerance = 1e-9;
   @Getter
   @Setter
   private int reportInterval = 0;
   @Getter
   @Setter
   private boolean cacheData = true;

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      GeneralizedLinearModel model = new GeneralizedLinearModel(this);
      Optimizer optimizer = (batchSize > 0)
                            ? BatchOptimizer.builder().batchSize(batchSize).subOptimizer(new com.davidbracewell.apollo.ml.classification.nn.SGD()).build()
                            : new SGD();
      WeightMatrix theta = new WeightMatrix(model.numberOfLabels(), model.numberOfFeatures());
      model.weights = optimizer.optimize(theta,
                                         dataset.vectorStream(cacheData),
                                         new GradientDescentCostFunction(loss, activation),
                                         TerminationCriteria.create()
                                                            .maxIterations(maxIterations)
                                                            .historySize(3)
                                                            .tolerance(tolerance),
                                         learningRate,
                                         weightUpdater,
                                         reportInterval).getWeights();
      model.activation = activation;
      return model;
   }
}// END OF SoftmaxLearner
