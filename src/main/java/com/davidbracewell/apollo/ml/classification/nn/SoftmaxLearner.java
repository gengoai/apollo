package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.classification.LinearModel;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.optimization.*;
import com.davidbracewell.apollo.ml.optimization.activation.Activation;
import com.davidbracewell.apollo.ml.optimization.loss.CrossEntropyLoss;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class SoftmaxLearner extends ClassifierLearner {
   @Getter
   @Setter
   private WeightUpdate weightUpdater = SGDUpdater.builder().build();
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
   private int reportInterval = 10;
   @Getter
   @Setter
   private boolean cacheData = true;

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      LinearModel model = new LinearModel(this);
      GradientDescentOptimizer optimizer = GradientDescentOptimizer.builder().batchSize(batchSize).build();
      model.weights = NDArrayFactory.defaultFactory().rand(model.numberOfLabels(), model.numberOfFeatures());
      model.bias = NDArrayFactory.defaultFactory().zeros(model.numberOfLabels());
      model.activation = Activation.SOFTMAX;
      optimizer.optimize(model,
                         dataset.vectorStream(false),
                         new GradientDescentCostFunction(new CrossEntropyLoss()),
                         TerminationCriteria.create()
                                            .maxIterations(maxIterations)
                                            .historySize(3)
                                            .tolerance(tolerance),
                         weightUpdater,
                         reportInterval);
      return model;
   }
}// END OF SoftmaxLearner
