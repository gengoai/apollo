package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.Instance;
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

   public boolean isVerbose() {
      return reportInterval > 0;
   }


   public void setVerbose(boolean verbose) {
      if (verbose) {
         this.reportInterval = 10;
      } else {
         this.reportInterval = 0;
      }
   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      int numL = dataset.getLabelEncoder().size();
      if (numL <= 2) {
         numL = 1;
      }
      LinearModel model = new LinearModel(this);
      GradientDescentOptimizer optimizer = GradientDescentOptimizer.builder().batchSize(batchSize).build();
      model.weights = NDArrayFactory.DEFAULT().rand(numL, model.numberOfFeatures());
      model.bias = NDArrayFactory.DEFAULT().zeros(numL);
      model.activation = Activation.SOFTMAX;
      optimizer.optimize(model,
                         dataset.vectorStream(cacheData),
                         new GradientDescentCostFunction(new CrossEntropyLoss(), numL > 1 ? -1 : 1),
                         TerminationCriteria.create()
                                            .maxIterations(maxIterations)
                                            .historySize(3)
                                            .tolerance(tolerance),
                         weightUpdater,
                         reportInterval);
      return model;
   }
}// END OF SoftmaxLearner
