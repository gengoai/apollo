package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.stream.MStream;

import java.util.Iterator;

/**
 * Mini-Batch Stochastic Gradient Descent Optimization
 *
 * @author David B. Bracewell
 */
public class GradientDescentOptimizer implements Optimizer<LinearModelParameters> {
   private final int batchSize;
   private double cost;

   /**
    * Instantiates a new Gradient descent optimizer.
    *
    * @param batchSize the batch size
    */
   public GradientDescentOptimizer(int batchSize) {
      this.cost = Double.POSITIVE_INFINITY;
      this.batchSize = batchSize;
   }


   @Override
   public double getFinalCost() {
      return cost;
   }

   @Override
   public void optimize(LinearModelParameters startingTheta,
                        MStream<NDArray> stream,
                        CostFunction<LinearModelParameters> costFunction,
                        StoppingCriteria stoppingCriteria,
                        WeightUpdate weightUpdater
                       ) {
      final BatchIterator iterator = new BatchIterator(stream.collect(),
                                                       startingTheta.numberOfLabels(),
                                                       startingTheta.numberOfFeatures());
      stoppingCriteria.untilTermination(iteration -> {
         cost = 0;
         iterator.shuffle();
         for (Iterator<NDArray> batch = iterator.iterator(batchSize); batch.hasNext(); ) {
            NDArray input = batch.next();
            CostGradientTuple cgt = costFunction.evaluate(input, startingTheta);
            cost += cgt.getCost() + weightUpdater.update(startingTheta, cgt.getGradient(), iteration);
         }
         cost /= iterator.size();
         return cost;
      });
   }

   @Override
   public void reset() {
      cost = Double.POSITIVE_INFINITY;
   }

}// END OF SGD
