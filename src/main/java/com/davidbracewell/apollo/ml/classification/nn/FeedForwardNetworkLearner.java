package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
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
   private LossFunction lossFunction = new LogLoss();
   @Getter
   @Setter
   @Builder.Default
   private WeightUpdate weightUpdate = new DeltaRule();
   @Getter
   @Setter
   @Builder.Default
   private double tolerance = 1e-4;
   @Getter
   @Setter
   @Builder.Default
   private int reportInterval = 10;
   @Getter
   @Setter
   @Builder.Default
   private int batchSize = 50;
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

   private List<CostGradientTuple> evaluate(FeedForwardNetwork network, Vector input) {
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
      Vector[] deltas = new Vector[numLayers + 1];

      deltas[numLayers] = lossFunction.derivative(predicted, y);
      for (int i = numLayers - 1; i >= 0; i--) {
         deltas[i] = network.layers.get(i).backward(activations[i], deltas[i + 1]);
      }

      List<CostGradientTuple> gradients = new ArrayList<>();
      for (int i = 0; i < numLayers; i++) {
         Vector a = i == 0 ? input : activations[i - 1];
         if (network.layers.get(i).hasWeights()) {
            GradientMatrix gm = GradientMatrix.calculate(a, deltas[i + 1]);
            gradients.add(CostGradientTuple.of(totalError, gm));
         }
      }

      return gradients;
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
      for (int i = 0; i < terminationCriteria.maxIterations(); i++) {
         Stopwatch sw = Stopwatch.createStarted();
         double loss = 0d;
         Collections.shuffle(vectors);
         for (int b = 0; b < vectors.size(); b += effectiveBatchSize) {
            List<CostGradientTuple> gradients = null;
            int count = 0;
            for (Vector datum : vectors.subList(b, Math.min(b + batchSize, vectors.size()))) {
               List<CostGradientTuple> c = evaluate(network, datum);
               loss += c.get(0).getCost();
               if (gradients == null) {
                  gradients = c;
               } else {
                  for (int j = 0; j < gradients.size(); j++) {
                     gradients.get(j).getGradient().add(c.get(j).getGradient());
                  }
               }
               count++;
            }
            numProcessed += count;
            lr = learningRate.get(lr, i, numProcessed);
            if (gradients != null) {
               for (CostGradientTuple gradient : gradients) {
                  gradient.getGradient().scale(1d / batchSize);
               }
               int index = 0;
               for (Layer layer : network.layers) {
                  if (layer.hasWeights()) {
                     weightUpdate.update(layer.getWeights(), gradients.get(index).getGradient(), lr, i);
                     index++;
                  }
               }
            }
         }
         sw.stop();
         if (reportInterval > 0 && ((i + 1) == terminationCriteria.maxIterations() || (i + 1) % reportInterval == 0)) {
            logInfo("iteration={0}, totalLoss={1}, time={2}", (i + 1), loss, sw);
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
