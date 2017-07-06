package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.nn.Layer.LayerBuilder;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.update.DeltaRule;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.Singular;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
@Builder
public class FeedForwardNetworkLearner extends ClassifierLearner {
   @Getter
   @Setter
   @Singular
   private List<LayerBuilder> layers;
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
   private boolean verbose = false;
   @Getter
   @Setter
   @Builder.Default
   private Optimizer optimizer = new SGD();


   private WeightComponent buildNetwork(FeedForwardNetwork network, int numFeatures, int numLabels) {
      int inputSize = numFeatures;
      network.layers = new com.davidbracewell.apollo.ml.nn.Layer[layers.size()];
      List<Weights> weights = new ArrayList<>();
      layers.get(layers.size() - 1).outputSize(numLabels);
      for (int i = 0; i < layers.size(); i++) {
         LayerBuilder layer = layers.get(i);
         network.layers[i] = layer.inputSize(inputSize).build();
         if (network.layers[i].hasWeights()) {
            weights.add(network.layers[i].getWeights());
         }
         inputSize = layer.getOutputSize();
      }
      return new WeightComponent(weights);
   }

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      FeedForwardNetwork network = new FeedForwardNetwork(this);
      WeightComponent theta = buildNetwork(network, network.numberOfFeatures(), network.numberOfLabels());
      CostWeightTuple optimal = optimizer.optimize(theta,
                                                   dataset::asVectors,
                                                   new BackpropagationCostFunction(network, lossFunction),
                                                   TerminationCriteria.create().maxIterations(maxIterations)
                                                                      .historySize(3)
                                                                      .tolerance(tolerance),
                                                   learningRate,
                                                   weightUpdate,
                                                   verbose
                                                  );
      for (int i = 0, index = 0; i < layers.size(); i++) {
         if (network.layers[i].hasWeights()) {
            network.layers[i].setWeights(optimal.getComponents().get(index));
            index++;
         }
      }
      return network;
   }

}// END OF FFlearner
