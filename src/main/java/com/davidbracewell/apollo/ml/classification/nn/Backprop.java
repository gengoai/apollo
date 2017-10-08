package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.optimization.*;
import com.davidbracewell.apollo.ml.optimization.loss.CrossEntropyLoss;
import com.davidbracewell.apollo.ml.optimization.loss.LossFunction;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.tuple.Tuple2;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class Backprop implements Optimizer<FeedForwardNetwork> {
   private double loss = 0d;
   @Getter
   @Setter
   private int batchSize = 32;
   @Getter
   @Setter
   private WeightUpdate weightUpdate = SGDUpdater.builder().build();
   @Getter
   @Setter
   private LossFunction lossFunction = new CrossEntropyLoss();

   @Override
   public double getFinalCost() {
      return loss;
   }

   @Override
   public void optimize(FeedForwardNetwork startingTheta,
                        SerializableSupplier<MStream<NDArray>> stream,
                        CostFunction<FeedForwardNetwork> costFunction,
                        TerminationCriteria terminationCriteria,
                        int reportInterval
                       ) {

      BatchIterator data = new BatchIterator(stream.get().collect(),
                                             startingTheta.numberOfLabels(),
                                             startingTheta.numberOfFeatures());

      WeightUpdate[] layerUpdates = new WeightUpdate[startingTheta.layers.size()];
      for (int i = 0; i < layerUpdates.length; i++) {
         layerUpdates[i] = weightUpdate.copy();
      }

      List<Layer> layers = startingTheta.layers;
      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
         loss = 0d;
         data.shuffle();
         for (Iterator<NDArray> itr = data.iterator(batchSize); itr.hasNext(); ) {
            NDArray X = itr.next();
            NDArray Y = X.getLabelAsNDArray();

            List<NDArray> ai = new ArrayList<>();
            NDArray cai = X;
            for (Layer layer : startingTheta.layers) {
               cai = layer.forward(cai);
               ai.add(cai);
            }
            loss += lossFunction.loss(cai, Y) / X.numCols();

            NDArray dz = lossFunction.derivative(cai, Y);
            for (int i = layers.size() - 1; i >= 0; i--) {
               NDArray input = i == 0 ? X : ai.get(i - 1);
               Tuple2<NDArray, Double> t = layers.get(i).backward(layerUpdates[i], input, ai.get(i), dz, iteration,
                                                                  i > 0);
               dz = t.v1;
               if (i == layers.size() - 1) {
                  loss += t.v2 / X.numCols();
               }
            }

         }
         boolean converged = terminationCriteria.check(loss);
         report(reportInterval, iteration, terminationCriteria.maxIterations(), converged, loss);
         if (converged) {
            break;
         }
      }

   }

   @Override
   public void reset() {
      loss = 0;
   }
}// END OF Backprop
