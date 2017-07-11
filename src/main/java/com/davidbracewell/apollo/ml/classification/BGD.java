package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.BottouLearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.alt.*;

/**
 * @author David B. Bracewell
 */
public class BGD extends BinaryClassifierLearner {
   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainForLabel(Dataset<Instance> dataset, double trueLabel) {
      BinaryGLM glm = new BinaryGLM(this);
      glm.activation = Activation.SIGMOID;
      SGD sgd = new SGD();
      CostWeightTuple tuple = sgd.optimize(
         new WeightVector(glm.numberOfFeatures()),
         dataset.asVectors()
                .map(fv -> {
                   if (fv.getLabelAsDouble() == trueLabel) {
                      fv.setLabel(1);
                   } else {
                      fv.setLabel(0);
                   }
                   return fv;
                })
                .cache(),
         new GradientDescentCostFunction(new LogLoss(), Activation.SIGMOID),
         TerminationCriteria.create().maxIterations(200).tolerance(1e-4).historySize(3),
         new BottouLearningRate(),
         new DeltaRule(),
         false
                                          );
      glm.bias = tuple.getWeights().getBias();
      glm.weights = tuple.getWeights().getWeights();
      return glm;
   }

}//END OF BGD
