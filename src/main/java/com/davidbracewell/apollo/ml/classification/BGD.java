package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.ConstantLearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.alt.*;

/**
 * @author David B. Bracewell
 */
public class BGD extends ClassifierLearner {
   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      BinaryGLM glm = new BinaryGLM(this);
      glm.activation = Activation.SIGMOID;
      SGD sgd = new SGD();
      CostWeightTuple tuple = sgd.optimize(
         new WeightVector(glm.numberOfFeatures()),
         dataset.asVectors().cache(),
         new GradientDescentCostFunction(new LogLoss(), Activation.SIGMOID),
         TerminationCriteria.create().maxIterations(20).tolerance(1e-4).historySize(3),
         new ConstantLearningRate(1.0),
         new DeltaRule(),
         false
                                          );
      glm.bias = tuple.getWeights().getBias();
      glm.weights = tuple.getWeights().getWeights();
      return glm;
   }
}//END OF BGD
