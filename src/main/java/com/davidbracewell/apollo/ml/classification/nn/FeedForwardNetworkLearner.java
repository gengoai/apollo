package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.loss.CrossEntropyLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.update.DeltaRule;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.guava.common.base.Stopwatch;
import com.davidbracewell.logging.Loggable;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.Singular;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

/**
 * @author David B. Bracewell
 */
@Builder
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
   private LearningRate learningRate = new BottouLearningRate();
   @Getter
   @Setter
   @Builder.Default
   private LossFunction lossFunction = new CrossEntropyLoss();
   @Getter
   @Setter
   @Builder.Default
   private WeightUpdate weightUpdate = new DeltaRule();
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
   private int batchSize = 20;
   @Getter
   @Setter
   @Builder.Default
   private Optimizer optimizer = new SGD();


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

   private double evaluate(FeedForwardNetwork network, Vector input) {
      Vector y = input.getLabelVector(network.numberOfLabels());
      int numLayers = network.layers.size();

      Vector[] activations = new Vector[numLayers];
      for (int i = 0; i < numLayers; i++) {
         if (i == 0) {
            activations[i] = network.layers.get(i).forward(input);
         } else {
            activations[i] = network.layers.get(i).forward(activations[i - 1]);
         }
      }
      Vector predicted = activations[activations.length - 1];
      double totalError = lossFunction.loss(predicted, y);

      Vector cDelta = lossFunction.derivative(predicted, y);
      for (int i = numLayers - 1; i >= 0; i--) {
         Vector inputVector = i == 0 ? input : activations[i - 1];
         cDelta = network.layers.get(i).backward(inputVector, activations[i], cDelta);
      }
      return totalError;
   }

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      FeedForwardNetwork network = new FeedForwardNetwork(this);
      buildNetwork(network, network.numberOfFeatures(), network.numberOfLabels());
      TerminationCriteria terminationCriteria = TerminationCriteria.create()
                                                                   .maxIterations(maxIterations)
                                                                   .tolerance(tolerance)
                                                                   .historySize(3);
      double lastLoss = 0;
      double lr = learningRate.getInitialRate();
      int numProcessed = 0;
      final int effectiveBatchSize = batchSize <= 0 ? 1 : batchSize;
      List<Vector> vectors = dataset.asVectors().collect();
      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
         Stopwatch sw = Stopwatch.createStarted();
         double loss = 0d;
         Collections.shuffle(vectors);
         for (int b = 0; b < vectors.size(); b += effectiveBatchSize) {
            int count = Math.min(b + effectiveBatchSize, vectors.size()) - b;
            if (count == 0) {
               continue;
            }
            loss += IntStream.range(b, Math.min(b + effectiveBatchSize, vectors.size()))
                             .mapToDouble(j -> evaluate(network, vectors.get(j)))
                             .sum();
            numProcessed += count;
            lr = learningRate.get(lr, iteration, numProcessed);
            if (count > 0) {
               for (Layer layer : network.layers) {
                  if (layer.hasWeights()) {
                     if (count > 1) {
                        layer.getGradient().scale(1d / count);
                     }
                     loss += weightUpdate.update(layer.getWeights(), layer.getGradient(), lr, iteration);
                     layer.setGradient(null);
                  }
               }
            }

         }
         sw.stop();
         if (reportInterval > 0 &&
                (iteration == 0 || (iteration + 1) == terminationCriteria.maxIterations() || (iteration + 1) % reportInterval == 0)) {
            logInfo("iteration={0}, totalLoss={1}, time={2}", (iteration + 1), loss, sw);
         }
         lastLoss = loss;
         if (terminationCriteria.check(lastLoss)) {
            break;
         }
      }

//      network.layers.removeIf(layer -> layer instanceof TrainOnlyLayer);
      network.layers.trimToSize();
      return network;
   }

}// END OF FFlearner
