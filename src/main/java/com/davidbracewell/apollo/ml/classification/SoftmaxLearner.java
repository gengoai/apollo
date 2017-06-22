package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxActivation;
import com.davidbracewell.apollo.optimization.loss.CrossEntropyLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.regularization.DeltaRule;
import com.davidbracewell.apollo.optimization.regularization.Regularizer;
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
   private Regularizer weightUpdater = new DeltaRule();
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
   private boolean verbose = false;

   private CostGradientTuple observe(Vector next, Weights weights) {
      return loss.lossAndDerivative(activation.apply(weights.dot(next)), next.getLabelVector(weights.numClasses()));
   }


   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      GLM model = new GLM(this);
      OnlineOptimizer optimizer;
      if (batchSize > 1) {
         optimizer = new OnlineBatchOptimizer(new StochasticGradientDescent(), batchSize);
      } else {
         optimizer = new StochasticGradientDescent();
      }

      Weights start = Weights.multiClass(model.numberOfLabels(), model.numberOfFeatures());
      Weights weights = optimizer.optimize(start,
                                           dataset::asVectors,
                                           this::observe,
                                           TerminationCriteria.create()
                                                              .maxIterations(maxIterations)
                                                              .historySize(3)
                                                              .tolerance(tolerance),
                                           learningRate,
                                           weightUpdater,
                                           verbose).getWeights();
      model.weights = weights;
      model.activation = activation;
      return model;
   }
}// END OF SoftmaxLearner
