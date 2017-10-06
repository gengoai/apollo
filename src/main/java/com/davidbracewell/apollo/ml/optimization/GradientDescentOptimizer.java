package com.davidbracewell.apollo.ml.optimization;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
@Builder
public class GradientDescentOptimizer implements Optimizer<LinearModelParameters> {
   double cost = Double.POSITIVE_INFINITY;
   @Getter
   @Setter
   @Builder.Default
   WeightUpdate weightUpdater = SGDUpdater.builder().build();
   @Getter
   @Setter
   @Builder.Default
   int maxIterations = 100;
   @Getter
   @Setter
   @Builder.Default
   int batchSize = 32;
   @Getter
   @Setter
   @Builder.Default
   double tolerance = 1e-6;

   @Override
   public double getFinalCost() {
      return cost;
   }

   @Override
   public void optimize(LinearModelParameters startingTheta,
                        SerializableSupplier<MStream<NDArray>> stream,
                        CostFunction<LinearModelParameters> costFunction,
                        TerminationCriteria terminationCriteria,
                        int reportInterval
                       ) {
      BatchIterator iterator = new BatchIterator(stream.get().collect(), 3, 4);
      TerminationCriteria tc = TerminationCriteria.create().maxIterations(maxIterations)
                                                  .tolerance(tolerance);
      for (int iteration = 0; iteration < tc.maxIterations(); iteration++) {
         cost = 0;
         for (Iterator<NDArray> batch = iterator.iterator(batchSize); batch.hasNext(); ) {
            NDArray input = batch.next();
            CostGradientTuple cgt = costFunction.evaluate(input, startingTheta);
            cost += cgt.getCost() + weightUpdater.update(startingTheta, cgt.getGradient(), iteration);
         }
         boolean converged = tc.check(cost);
         report(reportInterval, iteration, tc.maxIterations(), converged, cost);
         if (converged) {
            break;
         }
      }
   }


   @Override
   public void reset() {
      cost = Double.POSITIVE_INFINITY;
      weightUpdater.reset();
   }
}// END OF SGD
