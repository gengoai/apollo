package com.gengoai.apollo.ml.neural;

import com.gengoai.Stopwatch;
import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.optimization.*;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.math.NumericComparison;
import com.gengoai.stream.MStream;
import com.gengoai.tuple.Tuple2;
import com.gengoai.tuple.Tuple3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class Backprop implements Optimizer<FeedForwardNetwork> {
   private double loss = 0d;
   private int batchSize = 32;
   private int threads = 4;

   static double correct(NDArray predicted, NDArray gold) {
      NDArray pMax = predicted.argMax(Axis.COLUMN);
      NDArray gMax = gold.argMax(Axis.COLUMN);
      return pMax.testi(gMax, NumericComparison.EQ).scalarSum();
   }

   @Override
   public double getFinalCost() {
      return loss;
   }

   @Override
   public void optimize(FeedForwardNetwork startingTheta,
                        SerializableSupplier<MStream<NDArray>> stream,
                        CostFunction<FeedForwardNetwork> costFunction,
                        TerminationCriteria terminationCriteria,
                        WeightUpdate weightUpdate,
                        int reportInterval
                       ) {


      BatchIterator data = new BatchIterator(stream.get().collect(),
                                             startingTheta.getNumberOfLabels(),
                                             startingTheta.getNumberOfFeatures());

      WeightUpdate[] layerUpdates = new WeightUpdate[startingTheta.layers.size()];
      for (int i = 0; i < layerUpdates.length; i++) {
         layerUpdates[i] = weightUpdate.copy();
      }
//
//      ExecutorService executor = Executors.newFixedThreadPool(threads);
      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
         loss = 0d;
//         data.shuffle();
//         List<Future<Tuple3<Double, Double, List<Layer>>>> futures = new ArrayList<>();
//         for (Iterator<NDArray> itr = data.iterator(batchSize); itr.hasNext(); ) {
//            layerUpdates = new WeightUpdate[startingTheta.layers.size()];
//            for (int i = 0; i < layerUpdates.length; i++) {
//               layerUpdates[i] = weightUpdate.copy();
//            }
//            WThread wt = new WThread(itr.next(),
//                                     startingTheta.copy(),
//                                     costFunction,
//                                     Arrays.asList(layerUpdates),
//                                     iteration
//            );
//            futures.add(executor.submit(wt));
//         }
//
//         futures.forEach(f -> {
//            try {
//               Tuple3<Double, Double, List<Layer>> t3 = f.get();
//               for (int i = 0; i < t3.v3.size(); i++) {
//                  startingTheta.layers.get(i).update(
//                     new NDArray[]{t3.v3.get(i).getWeights()},
//                     new NDArray[]{t3.v3.get(i).getBias()}
//                                                    );
//               }
//            } catch (Exception e) {
//               e.printStackTrace();
//            }
//         });

         Stopwatch timer = Stopwatch.createStarted();
         List<Layer> layers = startingTheta.layers;
         for (Iterator<NDArray> itr = data.iterator(batchSize); itr.hasNext(); ) {
            NDArray X = itr.next();
            CostGradientTuple cgt = costFunction.evaluate(X, startingTheta);
            List<NDArray> ai = Arrays.asList(cgt.getActivations());
            NDArray dz = cgt.getGradient().getWeightGradient();
            loss += cgt.getCost();
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
         if (report(reportInterval, iteration, terminationCriteria, loss, timer.toString())) {
            break;
         }
      }

   }

   @Override
   public void reset() {
      loss = 0;
   }

   public static class WThread implements Callable<Tuple3<Double, Double, List<Layer>>> {
      final List<WeightUpdate> weightUpdates;
      final int iteration;
      public double loss;
      public double correct;
      public List<Layer> layers = new ArrayList<>();
      public FeedForwardNetwork network;
      public NDArray datum;
      CostFunction<FeedForwardNetwork> costFunction;

      public WThread(NDArray data,
                     FeedForwardNetwork network,
                     CostFunction<FeedForwardNetwork> costFunction,
                     List<WeightUpdate> weightUpdates,
                     int iteration
                    ) {
         for (Layer layer : network.layers) {
            layers.add(layer.copy());
         }
         this.network = network;
         this.datum = data;
         this.costFunction = costFunction;
         this.weightUpdates = weightUpdates;
         this.iteration = iteration;
      }

      @Override
      public Tuple3<Double, Double, List<Layer>> call() {
         double size = 0;
         NDArray X = datum;
         NDArray Y = datum.getLabelAsNDArray();
         double bSize = X.numCols();
         size += bSize;
         CostGradientTuple cgt = costFunction.evaluate(X, network);
         loss += cgt.getCost();
         List<NDArray> ai = Arrays.asList(cgt.getActivations());
         correct += correct(cgt.getActivations()[cgt.getActivations().length - 1], Y);
         NDArray dz = cgt.getGradient().getWeightGradient();
         for (int i = layers.size() - 1; i >= 0; i--) {
            NDArray input = i == 0 ? X : ai.get(i - 1);
            Tuple2<NDArray, Double> t = layers.get(i).backward(weightUpdates.get(i),
                                                               input,
                                                               ai.get(i),
                                                               dz,
                                                               iteration,
                                                               i > 0);
            dz = t.v1;
            if (i == layers.size() - 1) {
               loss += t.v2;
            }
         }
         loss = size > 0 ? loss / size : loss;
         return $(loss, correct, layers);
      }
   }
}// END OF Backprop
