package com.davidbracewell.apollo.optimization.o2;

import com.davidbracewell.apollo.optimization.CostWeightTuple;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.logging.Loggable;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class SGD implements Optimizer, Serializable, Loggable {
   private static final long serialVersionUID = 1L;

   @Override
   public CostWeightTuple optimize(WeightComponent initialTheta, CostFunction costFunction,
                                   TerminationCriteria terminationCriteria, LearningRate learningRate,
                                   WeightUpdate weightUpdater, boolean verbose
                                  ) {


      WeightComponent weights = initialTheta;
      double lr = learningRate.getInitialRate();
      double lastLoss = 0;
      int numProcessed = 0;
      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
         final double eta = learningRate.get(lr, 0, numProcessed);
         lr = eta;
         double sumTotal = 0;



         if (verbose && iteration % 10 == 0) {
            logInfo("pass={0}, total_cost={1}", iteration, sumTotal);
         }
         lastLoss = sumTotal;
         if (terminationCriteria.check(sumTotal)) {
            break;
         }
      }

      return CostWeightTuple.of(lastLoss, null);
   }
}//END OF SGD
