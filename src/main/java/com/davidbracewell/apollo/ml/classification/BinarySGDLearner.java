package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SigmoidFunction;
import com.davidbracewell.apollo.optimization.activation.StepFunction;
import com.davidbracewell.apollo.optimization.loss.HingeLoss;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.regularization.NonRegularizedDeltaRule;
import com.davidbracewell.apollo.optimization.regularization.WeightUpdater;
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
   private LearningRate learningRate = new ConstantLearningRate(0.1);
   @Getter
   @Setter
   private WeightUpdater weightUpdater = new NonRegularizedDeltaRule();
   @Getter
   @Setter
   private LossFunction loss = new LogLoss();
   @Getter
   @Setter
   private Activation activation = new SigmoidFunction();
   @Getter
   @Setter
   private int maxIterations = 300;
   @Getter
   @Setter
   private int batchSize = -1;
   @Getter
   @Setter
   private double tolerance = 1e-9;
   @Getter
   @Setter
   private boolean verbose = false;

   /**
    * Linear svm binary sgd learner.
    *
    * @return the binary sgd learner
    */
   public static BinarySGDLearner linearSVM() {
      BinarySGDLearner learner = new BinarySGDLearner();
      learner.setActivation(new StepFunction());
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
      learner.setActivation(new StepFunction());
      learner.setLoss(new HingeLoss(0.0));
      learner.setLearningRate(new ConstantLearningRate(1.0));
      return learner;
   }

   private LossGradientTuple observe(Vector next, Weights weights) {
      return loss.lossAndDerivative(activation.apply(weights.dot(next)), next.getLabelVector(weights.numClasses()));
   }

   @Override
   public void reset() {

   }

   @Override
   protected Classifier trainForLabel(Dataset<Instance> dataset, double trueLabel) {
      Optimizer optimizer;
      if (batchSize > 1) {
         optimizer = new BatchOptimizer(new SGD(), batchSize);
      } else {
         optimizer = new SGD();
      }

      Weights start;
      start = Weights.binary(dataset.getFeatureEncoder().size());

      Weights weights = optimizer.optimize(start,
                                           () -> dataset.asFeatureVectors()
                                                        .map(fv -> {
                                                           if (fv.getLabel() == trueLabel) {
                                                              fv.setLabel(1);
                                                           } else {
                                                              fv.setLabel(0);
                                                           }
                                                           return fv;
                                                        })
                                                        .cache(),
                                           this::observe,
                                           TerminationCriteria.create()
                                                              .maxIterations(maxIterations)
                                                              .historySize(3)
                                                              .tolerance(tolerance),
                                           learningRate,
                                           weightUpdater,
                                           verbose).getWeights();

      BinaryGLM glm = new BinaryGLM(dataset.getEncoderPair(), dataset.getPreprocessors());
      glm.weights = weights.getTheta().row(0).copy();
      glm.bias = weights.getBias().get(0);
      return glm;
   }

}// END OF SGDLearner