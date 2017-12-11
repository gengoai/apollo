package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.optimization.*;
import com.davidbracewell.apollo.ml.optimization.activation.Activation;
import com.davidbracewell.apollo.ml.optimization.activation.SignActivation;
import com.davidbracewell.apollo.ml.optimization.loss.HingeLoss;
import com.davidbracewell.apollo.ml.optimization.loss.LogLoss;
import com.davidbracewell.apollo.ml.optimization.loss.LossFunction;
import com.davidbracewell.logging.Loggable;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
@NoArgsConstructor
@AllArgsConstructor
public class BinaryGradientDescentLearner extends BinaryClassifierLearner implements Loggable {
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
   private LossFunction lossFunction = new LogLoss();
   @Getter
   @Setter
   private Activation activation = new SignActivation();
   @Getter
   @Setter
   private WeightUpdate weightUpdater = SGDUpdater.builder().build();

   @Override
   protected void resetLearnerParameters() {

   }

   public static BinaryGradientDescentLearner logisticRegression() {
      BinaryGradientDescentLearner learner = new BinaryGradientDescentLearner();
      learner.setLossFunction(new LogLoss());
      learner.setActivation(Activation.SIGMOID);
      return learner;
   }

   public static BinaryGradientDescentLearner linearSVM() {
      BinaryGradientDescentLearner learner = new BinaryGradientDescentLearner();
      learner.setLossFunction(new HingeLoss());
      learner.setActivation(new SignActivation());
      return learner;
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


   public void setOptimizer(String optimizer) {
      switch (optimizer.toLowerCase()) {
         case "sgd":
            this.weightUpdater = SGDUpdater.builder().build();
            break;
         case "adam":
            this.weightUpdater = AdamUpdater.builder().build();
            break;
         default:
            throw new IllegalArgumentException("Unknown optimizer " + optimizer);
      }
   }

   @Override
   protected Classifier trainForLabel(Dataset<Instance> dataset, double trueLabel) {
      LinearModel model = new LinearModel(this, true);
      model.weights = NDArrayFactory.DEFAULT().rand(1, model.numberOfFeatures());
      model.bias = NDArrayFactory.DEFAULT().scalar(0);
      model.activation = activation;
      GradientDescentOptimizer optimizer = GradientDescentOptimizer.builder().batchSize(batchSize).build();
      optimizer.optimize(model,
                         dataset.vectorStream(true),
                         new GradientDescentCostFunction(lossFunction, (int) trueLabel),
                         TerminationCriteria.create()
                                            .maxIterations(maxIterations)
                                            .historySize(3)
                                            .tolerance(tolerance),
                         weightUpdater,
                         reportInterval);
      return model;
   }

}//END OF BinaryGradientDescentLearner
