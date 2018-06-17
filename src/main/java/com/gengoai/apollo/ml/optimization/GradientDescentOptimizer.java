package com.gengoai.apollo.ml.optimization;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
@Builder
public class GradientDescentOptimizer implements Optimizer<LinearModelParameters> {
   @Builder.Default
   double cost = Double.POSITIVE_INFINITY;
   @Getter
   @Setter
   @Builder.Default
   int batchSize = 32;

   @Override
   public double getFinalCost() {
      return cost;
   }

   @Override
   public void optimize(LinearModelParameters startingTheta,
                        SerializableSupplier<MStream<NDArray>> stream,
                        CostFunction<LinearModelParameters> costFunction,
                        TerminationCriteria terminationCriteria,
                        WeightUpdate weightUpdater,
                        int reportInterval
                       ) {
      BatchIterator iterator = new BatchIterator(stream.get().collect(),
                                                 startingTheta.numberOfLabels(),
                                                 startingTheta.numberOfFeatures());
      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
         cost = 0;
         iterator.shuffle();
//         val timer = Stopwatch.createStarted();
         for (Iterator<NDArray> batch = iterator.iterator(batchSize); batch.hasNext(); ) {
            NDArray input = batch.next();
            CostGradientTuple cgt = costFunction.evaluate(input, startingTheta);
            cost += cgt.getCost() + weightUpdater.update(startingTheta, cgt.getGradient(), iteration);
         }
         cost /= iterator.size();
//         timer.stop();
         if (report(reportInterval, iteration, terminationCriteria, cost, "timer.toString()")) {
            break;
         }
      }
   }

   @Override
   public void reset() {
      cost = Double.POSITIVE_INFINITY;
   }

}// END OF SGD
