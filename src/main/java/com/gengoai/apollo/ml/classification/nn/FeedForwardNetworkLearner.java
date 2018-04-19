package com.gengoai.apollo.ml.classification.nn;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.classification.Classifier;
import com.gengoai.apollo.ml.classification.ClassifierLearner;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.optimization.AdamUpdater;
import com.gengoai.apollo.ml.optimization.SGDUpdater;
import com.gengoai.apollo.ml.optimization.TerminationCriteria;
import com.gengoai.apollo.ml.optimization.WeightUpdate;
import com.gengoai.apollo.ml.optimization.loss.CrossEntropyLoss;
import com.gengoai.apollo.ml.optimization.loss.LossFunction;
import com.gengoai.mango.logging.Loggable;
import com.gengoai.mango.tuple.Tuple2;
import com.gengoai.mango.tuple.Tuple3;
import com.gengoai.apollo.ml.classification.Classifier;
import com.gengoai.apollo.ml.classification.ClassifierLearner;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.optimization.AdamUpdater;
import com.gengoai.apollo.ml.optimization.SGDUpdater;
import com.gengoai.apollo.ml.optimization.TerminationCriteria;
import com.gengoai.apollo.ml.optimization.WeightUpdate;
import com.gengoai.apollo.ml.optimization.loss.CrossEntropyLoss;
import com.gengoai.apollo.ml.optimization.loss.LossFunction;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.Singular;
import org.jblas.FloatMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import static com.gengoai.mango.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
@Builder(builderClassName = "NetworkBuilder")
public class FeedForwardNetworkLearner extends ClassifierLearner implements Loggable {
   @Getter
   @Setter
   @Singular
   private List<Layer.LayerBuilder> layers;
   @Getter
   @Setter
   @Builder.Default
   private int maxIterations = 200;
   @Getter
   @Setter
   @Builder.Default
   private LossFunction lossFunction = new CrossEntropyLoss();
   @Getter
   @Setter
   @Builder.Default
   private double tolerance = 1e-9;
   @Getter
   @Setter
   @Builder.Default
   private int reportInterval = 10;
   @Getter
   @Setter
   @Builder.Default
   private int batchSize = 100;
   @Getter
   @Setter
   @Builder.Default
   private WeightUpdate weightUpdate = SGDUpdater.builder().build();

   static float correct(FloatMatrix predicted, FloatMatrix gold) {
      int[] pMax = predicted.columnArgmaxs();
      int[] gMax = gold.columnArgmaxs();
      float correct = 0;
      for (int i = 0; i < pMax.length; i++) {
         if (pMax[i] == gMax[i]) {
            correct++;
         }
      }
      return correct;
   }

   private void buildNetwork(FeedForwardNetwork network, int numFeatures, int numLabels) {
      int inputSize = numFeatures;
      network.layers = new ArrayList<>();
      layers.get(layers.size() - 1).outputSize(numLabels);
      for (Layer.LayerBuilder layer : layers) {
         if (layer.getOutputSize() <= 0) {
            layer.outputSize(inputSize);
         }
         network.layers.add(layer.inputSize(inputSize).build());
         inputSize = layer.getOutputSize();
      }
   }

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      FeedForwardNetwork network = new FeedForwardNetwork(this);

      buildNetwork(network, network.numberOfFeatures(), network.numberOfLabels());

      for (Layer layer : network.layers) {
         layer.preTrain(dataset);
      }

      TerminationCriteria terminationCriteria = TerminationCriteria.create()
                                                                   .maxIterations(maxIterations)
                                                                   .tolerance(tolerance)
                                                                   .historySize(3);
      Backprop bp = new Backprop();
      bp.setBatchSize(batchSize);
      bp.optimize(network,
                  dataset.vectorStream(false),
                  new FeedForwardCostFunction(lossFunction),
                  terminationCriteria,
                  weightUpdate,
                  reportInterval);
      return network;
//
//      BatchIterator data = new BatchIterator(dataset);
//      List<List<WeightUpdate>> threadedWeightUpdates = new ArrayList<>();
//      for (int j = 0; j < 4; j++) {
//         threadedWeightUpdates.add(new ArrayList<>());
//      }
////      List<WeightUpdate> weightUpdates = new ArrayList<>();
//      for (int i = 0; i < network.layers.size(); i++) {
//         for (int j = 0; j < 4; j++) {
//            threadedWeightUpdates.get(j).add(weightUpdate.copy());
//         }
////         weightUpdates.add(weightUpdate.copy());
//      }
//      final ExecutorService executor = Executors.newFixedThreadPool(4);
//
//      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
//         Stopwatch timer = Stopwatch.createStarted();
//         double loss = 0d;
//         double correct = 0;
//         data.shuffle();
//         List<Future<Tuple3<Double, Double, List<Layer>>>> results = new ArrayList<>();
//         List<NDArray> batches = Lists.asArrayList(data.iterator(effectiveBatchSize));
//         int thread = 0;
//         int partitionSize = batches.size() / 4;
//         int start = 0;
//         for (int b = 0; b < 3; b++) {
//            results.add(executor.submit(new WThread(batches.subList(start, start + partitionSize),
//                                                    network,
//                                                    lossFunction,
//                                                    threadedWeightUpdates.get(b),
//                                                    iteration)));
//            start += partitionSize;
//         }
//         results.add(executor.submit(new WThread(batches.subList(start, batches.size()),
//                                                 network,
//                                                 lossFunction,
//                                                 threadedWeightUpdates.get(3),
//                                                 iteration)));
////         for (List<NDArray> tuple2s : com.gengoai.guava.common.collect.Lists.partition(batches,
////                                                                                              (int) Math.ceil(
////                                                                                                 batches.size() / 4))) {
////            results.add(executor.submit(
////               new WThread(tuple2s, network, lossFunction, threadedWeightUpdates.get(thread), iteration)));
////            thread++;
////         }
////         for (Iterator<Tuple2<NDArray, NDArray>> itr = data.iterator(effectiveBatchSize); itr.hasNext(); ) {
////            numBatch++;
////            results.add(executor.submit(new WorkerThread(itr.next(), network, lossFunction, weightUpdates, iteration)));
//////            Tuple2<NDArray, NDArray> tuple = itr.next();
//////            NDArray X = tuple.v1;
//////            NDArray Y = tuple.v2;
//////            double bSize = X.numCols();
//////            List<NDArray> ai = new ArrayList<>();
//////            NDArray cai = X;
//////            for (Layer layer : network.layers) {
//////               cai = layer.forward(cai);
//////               ai.add(cai);
//////            }
//////            loss += lossFunction.loss(cai, Y) / bSize;
//////            correct += correct(cai.toFloatMatrix(), Y.toFloatMatrix());
//////            NDArray dz = lossFunction.derivative(cai, Y);
//////            for (int i = network.layers.size() - 1; i >= 0; i--) {
//////               NDArray input = i == 0 ? X : ai.get(i - 1);
//////               Tuple2<NDArray, Double> gradCost = network.layers.get(i).backward(weightUpdates.get(i), input, ai.get(i),
//////                                                                                dz, iteration, i > 0);
//////               dz = gradCost.v1;
//////               loss += gradCost.v2;
//////            }
//////            numProcessed += bSize;
////         }
//
//         NDArray[][] wUpdates = new NDArray[network.layers.size()][results.size()];
//         NDArray[][] bUpdates = new NDArray[network.layers.size()][results.size()];
//         thread = 0;
//         for (Future<Tuple3<Double, Double, List<Layer>>> future : results) {
//            try {
//               Tuple3<Double, Double, List<Layer>> r = future.get();
//               loss += r.v1;
//               correct += r.v2;
//               List<Layer> layers = r.v3;
//               for (int i = 0; i < layers.size(); i++) {
//                  wUpdates[i][thread] = layers.get(i).getWeights();
//                  bUpdates[i][thread] = layers.get(i).getBias();
//               }
//               thread++;
//            } catch (InterruptedException | ExecutionException e) {
//               e.printStackTrace();
//            }
//         }
//         loss /= results.size();
//         for (int i = 0; i < layers.size(); i++) {
//            network.layers.get(i).update(wUpdates[i], bUpdates[i]);
//         }
//
////         NDArray[] wGrad = new NDArray[network.layers.size()];
////         NDArray[] bGrad = new NDArray[network.layers.size()];
////         for (Future<Tuple3<Double, Double, BackpropResult[]>> future : results) {
////            try {
////               Tuple3<Double, Double, BackpropResult[]> costAcc = future.get();
////               loss += costAcc.v1;
////               correct += costAcc.v2;
////               BackpropResult[] bp = costAcc.v3;
////               for (int i = 0; i < bp.length; i++) {
////                  if (wGrad[i] == null && !bp[i].getWeightGradient().isEmpty()) {
////                     wGrad[i] = bp[i].getWeightGradient();
////                  } else if (!bp[i].getWeightGradient().isEmpty()) {
////                     wGrad[i].addi(bp[i].getWeightGradient());
////                  }
////                  if (bGrad[i] == null && !bp[i].getBiasGradient().isEmpty()) {
////                     bGrad[i] = bp[i].getBiasGradient();
////                  } else if (!bp[i].getBiasGradient().isEmpty()) {
////                     bGrad[i].addi(bp[i].getBiasGradient());
////                  }
////               }
////            } catch (Exception e) {
////               e.printStackTrace();
////            }
////         }
////
////         for (int i = 0; i < wGrad.length; i++) {
////            if (wGrad[i] != null) {
////               wGrad[i].divi(data.size());
////            }
////            if (bGrad[i] != null) {
////               bGrad[i].divi(data.size());
////            }
////         }
////
////         for (int i = 0; i < network.layers.size(); i++) {
////            loss += network.layers.get(i).update(
////               weightUpdates.get(i),
////               wGrad[i],
////               bGrad[i],
////               iteration);
////         }
//
//         if (reportInterval > 0 &&
//                (iteration == 0 || (iteration + 1) == terminationCriteria.maxIterations() || (iteration + 1) % reportInterval == 0)) {
//            logInfo("iteration={0}, totalLoss={1}, accuracy={2}, time={3}",
//                    (iteration + 1),
//                    (loss),
//                    (correct / data.size()),
//                    timer);
//         }
//         if (terminationCriteria.check(loss)) {
//            break;
//         }
//
//      }
//      executor.shutdown();
//      network.layers.removeIf(Layer::trainOnly);
//      network.layers.trimToSize();
//      return network;
   }

   public static class WThread implements Callable<Tuple3<Double, Double, List<Layer>>> {
      final List<WeightUpdate> weightUpdates;
      final int iteration;
      public double loss;
      public double correct;
      public List<Layer> layers = new ArrayList<>();
      public List<NDArray> data = new ArrayList<>();
      public LossFunction lossFunction;

      public WThread(List<NDArray> data, FeedForwardNetwork network, LossFunction lossFunction, List<WeightUpdate> weightUpdates, int iteration) {
         for (Layer layer : network.layers) {
            layers.add(layer.copy());
         }
         this.data = data;
         this.lossFunction = lossFunction;
         this.weightUpdates = weightUpdates;
         this.iteration = iteration;
      }

      @Override
      public Tuple3<Double, Double, List<Layer>> call() {
         double size = 0;
         for (NDArray datum : data) {
            NDArray X = datum;
            NDArray Y = datum.getLabelAsNDArray();
            double bSize = X.numCols();
            size += bSize;
            List<NDArray> ai = new ArrayList<>();
            NDArray cai = X;
            for (Layer layer : layers) {
               cai = layer.forward(cai);
               ai.add(cai);
            }
            loss += lossFunction.loss(cai, Y);
            correct += correct(cai.toFloatMatrix(), Y.toFloatMatrix());
            NDArray dz = lossFunction.derivative(cai, Y);
            for (int i = layers.size() - 1; i >= 0; i--) {
               NDArray input = i == 0 ? X : ai.get(i - 1);
               Tuple2<NDArray, Double> t = layers.get(i).backward(weightUpdates.get(i), input, ai.get(i), dz, iteration,
                                                                  i > 0);
               dz = t.v1;
               if (i == layers.size() - 1) {
                  loss += t.v2;
               }
            }
         }
         loss = size > 0 ? loss / size : loss;
         return $(loss, correct, layers);
      }
   }

   public static class NetworkBuilder {

      public NetworkBuilder optimizer(String optimizer) {
         switch (optimizer.toLowerCase()) {
            case "sgd":
               this.weightUpdate(SGDUpdater.builder().build());
               break;
            case "adam":
               this.weightUpdate(AdamUpdater.builder().build());
               break;
            default:
               throw new IllegalArgumentException("Unknown optimizer " + optimizer);
         }
         return this;
      }

   }

}// END OF FeedForwardNetworkLearner
