package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.activation.StepActivation;
import com.davidbracewell.apollo.optimization.loss.HingeLoss;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.update.DeltaRule;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import lombok.Getter;
import lombok.Setter;

/**
 * The type Binary sgd learner.
 *
 * @author David B. Bracewell
 */
public class BinarySGDLearner extends BinaryClassifierLearner {
   @Getter
   @Setter
   private LearningRate learningRate = new BottouLearningRate(1.0, 0.9);
   @Getter
   @Setter
   private WeightUpdate weightUpdater = new DeltaRule();
   @Getter
   @Setter
   private LossFunction loss = new LogLoss();
   @Getter
   @Setter
   private Activation activation = new SigmoidActivation();
   @Getter
   @Setter
   private int maxIterations = 200;
   @Getter
   @Setter
   private int batchSize = -1;
   @Getter
   @Setter
   private double tolerance = 1e-4;
   @Getter
   @Setter
   private int reportInterval = 0;
   @Getter
   @Setter
   private boolean cacheData = true;

   /**
    * Linear svm binary sgd learner.
    *
    * @return the binary sgd learner
    */
   public static BinarySGDLearner linearSVM() {
      BinarySGDLearner learner = new BinarySGDLearner();
      learner.setActivation(new StepActivation());
      learner.setLoss(new HingeLoss(1.0));
      return learner;
   }

   /**
    * Logistic regression binary sgd learner.
    *
    * @return the binary sgd learner
    */
   public static BinarySGDLearner logisticRegression() {
      return new BinarySGDLearner();
   }

   /**
    * Perceptron binary sgd learner.
    *
    * @return the binary sgd learner
    */
   public static BinarySGDLearner perceptron() {
      BinarySGDLearner learner = new BinarySGDLearner();
      learner.setActivation(new StepActivation());
      learner.setLoss(new HingeLoss(0.0));
      learner.setLearningRate(new ConstantLearningRate(1.0));
      return learner;
   }

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainForLabel(Dataset<Instance> dataset, double trueLabel) {
      BinaryGLM model = new BinaryGLM(this);
      WeightMatrix theta = new WeightMatrix(2, model.numberOfFeatures());
      Optimizer optimizer = new SGD();

      CostWeightTuple result = optimizer.optimize(theta,
                                                  dataset.vectorStream(cacheData, trueLabel),
                                                  new GradientDescentCostFunction(loss, activation),
                                                  TerminationCriteria.create()
                                                                     .maxIterations(maxIterations)
                                                                     .historySize(3)
                                                                     .tolerance(tolerance),
                                                  learningRate,
                                                  weightUpdater,
                                                  reportInterval
                                                 );
      model.weights = result.getWeights().getWeightVector(0);
      model.bias = result.getWeights().getBias(0);
      model.activation = activation;
      return model;
   }

}// END OF SGDLearner
