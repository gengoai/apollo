package com.davidbracewell.apollo.ml.classification.nn.slt;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.loss.CrossEntropyLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.guava.common.base.Stopwatch;
import com.davidbracewell.logging.Loggable;
import com.davidbracewell.tuple.Tuple2;
import com.davidbracewell.tuple.Tuple3;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.Singular;
import org.jblas.FloatMatrix;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static com.davidbracewell.tuple.Tuples.$;

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
   private WeightUpdate weightUpdate = SGD.builder().build();

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
      MatrixTrainSet data = new MatrixTrainSet(dataset);
      buildNetwork(network, network.numberOfFeatures(), network.numberOfLabels());
      TerminationCriteria terminationCriteria = TerminationCriteria.create()
                                                                   .maxIterations(maxIterations)
                                                                   .tolerance(tolerance)
                                                                   .historySize(3);
      int numProcessed = 0;
      final int effectiveBatchSize = batchSize <= 0 ? 1 : batchSize;

      try {
         dataset.close();
      } catch (Exception e) {
         e.printStackTrace();
      }

      List<WeightUpdate> weightUpdates = new ArrayList<>();
      for (int i = 0; i < network.layers.size(); i++) {
         weightUpdates.add(weightUpdate.copy());
      }
      final ExecutorService executor = Executors.newFixedThreadPool(4);

      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
         Stopwatch timer = Stopwatch.createStarted();
         double loss = 0d;
         float numBatch = 0;
         double correct = 0;
         data.shuffle();
         List<Future<Tuple3<Double, Double, BackpropResult[]>>> results = new ArrayList<>();
         for (Iterator<Tuple2<Matrix, Matrix>> itr = data.iterator(effectiveBatchSize); itr.hasNext(); ) {
            numBatch++;
            results.add(executor.submit(new WorkerThread(itr.next(), network, lossFunction, weightUpdates, iteration)));
////            Tuple2<Matrix, Matrix> tuple = itr.next();
////            Matrix X = tuple.v1;
////            Matrix Y = tuple.v2;
////            double bSize = X.numCols();
////            List<Matrix> ai = new ArrayList<>();
////            Matrix cai = X;
////            for (Layer layer : network.layers) {
////               cai = layer.forward(cai);
////               ai.add(cai);
////            }
////            loss += lossFunction.loss(cai, Y) / bSize;
////            correct += correct(cai.toFloatMatrix(), Y.toFloatMatrix());
////            Matrix dz = lossFunction.derivative(cai, Y);
////            for (int i = network.layers.size() - 1; i >= 0; i--) {
////               Matrix input = i == 0 ? X : ai.get(i - 1);
////               Tuple2<Matrix, Double> gradCost = network.layers.get(i).backward(weightUpdates.get(i), input, ai.get(i),
////                                                                                dz, iteration, i > 0);
////               dz = gradCost.v1;
////               loss += gradCost.v2;
////            }
////            numProcessed += bSize;
         }
         Matrix[] wGrad = new Matrix[network.layers.size()];
         Matrix[] bGrad = new Matrix[network.layers.size()];
         for (Future<Tuple3<Double, Double, BackpropResult[]>> future : results) {
            try {
               Tuple3<Double, Double, BackpropResult[]> costAcc = future.get();
               loss += costAcc.v1;
               correct += costAcc.v2;
               BackpropResult[] bp = costAcc.v3;
               for (int i = 0; i < bp.length; i++) {
                  if (wGrad[i] == null && !bp[i].getWeightGradient().isEmpty()) {
                     wGrad[i] = bp[i].getWeightGradient();
                  } else if (!bp[i].getWeightGradient().isEmpty()) {
                     wGrad[i].addi(bp[i].getWeightGradient());
                  }
                  if (bGrad[i] == null && !bp[i].getBiasGradient().isEmpty()) {
                     bGrad[i] = bp[i].getBiasGradient();
                  } else if (!bp[i].getBiasGradient().isEmpty()) {
                     bGrad[i].addi(bp[i].getBiasGradient());
                  }
               }
            } catch (Exception e) {
               e.printStackTrace();
            }
         }

         for (int i = 0; i < wGrad.length; i++) {
            if (wGrad[i] != null) {
               wGrad[i].divi(data.size());
            }
            if (bGrad[i] != null) {
               bGrad[i].divi(data.size());
            }
         }

         for (int i = 0; i < network.layers.size(); i++) {
            loss += network.layers.get(i).update(
               weightUpdates.get(i),
               wGrad[i],
               bGrad[i],
               iteration);
         }

         numProcessed += data.size();
         if (reportInterval > 0 &&
                (iteration == 0 || (iteration + 1) == terminationCriteria.maxIterations() || (iteration + 1) % reportInterval == 0)) {
            logInfo("iteration={0}, totalLoss={1}, accuracy={2}, time={3}",
                    (iteration + 1),
                    (loss / numBatch),
                    (correct / data.size()),
                    timer);
         }
         if (terminationCriteria.check(loss)) {
            break;
         }

      }
      executor.shutdown();
      network.layers.removeIf(Layer::trainOnly);
      network.layers.trimToSize();
      return network;
   }

   public static class WorkerThread implements Callable<Tuple3<Double, Double, BackpropResult[]>> {
      final Matrix X;
      final Matrix Y;
      final FeedForwardNetwork network;
      final LossFunction lossFunction;
      final List<WeightUpdate> weightUpdates;
      final int iteration;

      public WorkerThread(Tuple2<Matrix, Matrix> xy, FeedForwardNetwork network, LossFunction lossFunction, List<WeightUpdate> weightUpdates, int iteration) {
         this.X = xy.v1;
         this.Y = xy.v2;
         this.network = network;
         this.lossFunction = lossFunction;
         this.weightUpdates = weightUpdates;
         this.iteration = iteration;
      }

      @Override
      public Tuple3<Double, Double, BackpropResult[]> call() throws Exception {
         double bSize = X.numCols();
         List<Matrix> ai = new ArrayList<>();
         Matrix cai = X;
         for (Layer layer : network.layers) {
            cai = layer.forward(cai);
            ai.add(cai);
         }
         double loss = lossFunction.loss(cai, Y) / bSize;
         double correct = correct(cai.toFloatMatrix(), Y.toFloatMatrix());
         Matrix dz = lossFunction.derivative(cai, Y);
         BackpropResult[] results = new BackpropResult[network.layers.size()];
         for (int i = network.layers.size() - 1; i >= 0; i--) {
            Matrix input = i == 0 ? X : ai.get(i - 1);
            results[i] = network.layers.get(i).backward(input, ai.get(i), dz, i > 0);
            dz = results[i].getDelta();
         }
//         synchronized (network) {
//            for (int i = network.layers.size() - 1; i >= 0; i--) {
//               Matrix input = i == 0 ? X : ai.get(i - 1);
//               Tuple2<Matrix, Double> gradCost = network.layers.get(i).backward(weightUpdates.get(i), input, ai.get(i),
//                                                                                dz, iteration, i > 0);
//               dz = gradCost.v1;
//               loss += gradCost.v2;
//            }
//         }
         return $(loss, correct, results);
      }


   }

   public static class NetworkBuilder {

      public NetworkBuilder optimizer(String optimizer) {
         switch (optimizer.toLowerCase()) {
            case "sgd":
               this.weightUpdate(SGD.builder().build());
               break;
            case "adam":
               this.weightUpdate(Adam.builder().build());
               break;
            default:
               throw new IllegalArgumentException("Unknown optimizer " + optimizer);
         }
         return this;
      }

   }

}// END OF FeedForwardNetworkLearner
