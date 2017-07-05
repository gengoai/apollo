package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxActivation;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.update.DeltaRule;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableConsumer;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.*;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author David B. Bracewell
 */
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FeedForwardLearner extends ClassifierLearner {
   @Getter
   private @Singular
   List<Layer> layers = new ArrayList<>();
   @Getter
   @Setter
   @Builder.Default
   private int maxIterations = 200;
   @Getter
   @Setter
   @Builder.Default
   private Activation outputActivation = new SoftmaxActivation();
   @Getter
   @Setter
   @Builder.Default
   private LearningRate learningRate = new BottouLearningRate(0.1, 0.001);
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
   @Getter
   @Setter
   @Builder.Default
   private int maxPreTrainIterations = 0;
   @Getter
   @Setter
   @Builder.Default
   private int preTrainBatchSize = 20;
   @Getter
   @Setter
   @Builder.Default
   private LearningRate preTrainLearningRate = new DecayLearningRate(0.1, 0.001);
   @Getter
   @Setter
   @Builder.Default
   private WeightInitializer weightInitializer = WeightInitializer.DEFAULT;
   private int numberOfWeightLayers = 0;
   @Getter
   @Setter
   @Builder.Default
   private SerializableConsumer<Layer> layerProcessor = (layer -> {});

   private void connect(Layer current, Layer previous) {
      current.connect(previous);
      if (current.hasWeights()) {
         numberOfWeightLayers++;
         weightInitializer.initialize(current.getWeights());
      }
   }

   private Layer[] makeLayers(int numberOfFeatures, int numberOfLabels) {
      List<Layer> layerList = new ArrayList<>(layers);
      layerList.add(new OutputLayer(outputActivation, numberOfLabels));
      InputLayer inputLayer = new InputLayer(numberOfFeatures);
      connect(layerList.get(0), inputLayer);
      for (int i = 1; i < layerList.size(); i++) {
         connect(layerList.get(i), layerList.get(i - 1));
      }
      return layerList.toArray(new Layer[layerList.size()]);
   }

   @Override
   protected void resetLearnerParameters() {
      numberOfWeightLayers = 0;
   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      FeedForwardNetwork model = new FeedForwardNetwork(this);

      model.layers = makeLayers(dataset.getFeatureEncoder().size(), dataset.getLabelEncoder().size());
      int numberOfLayers = model.layers.length;

      int[][] shapes = new int[numberOfWeightLayers][2];

      for (int i = 0, index = 0; i < numberOfLayers; i++) {
         if (model.layers[i].hasWeights()) {
            shapes[index][0] = model.layers[i].getOutputSize();
            shapes[index][1] = (i == 0) ? dataset.getFeatureEncoder().size() : model.layers[i - 1].getOutputSize();
            index++;
         }
      }
      WeightComponent theta = new WeightComponent(shapes, WeightInitializer.ZEROES);
      for (int i = 0, index = 0; i < numberOfLayers; i++) {
         if (model.layers[i].hasWeights()) {
            theta.set(index, model.layers[i].getWeights());
            index++;
         }
      }

      //Do pretraining
      AtomicDouble preTrainLR = new AtomicDouble(preTrainLearningRate.getInitialRate());
      AtomicInteger preTrainNumProcessed = new AtomicInteger(0);
      for (int preTrainIteration = 0; preTrainIteration < maxPreTrainIterations; preTrainIteration++) {
         final int iteration = preTrainIteration;
         dataset.shuffle().asVectors().partition(preTrainBatchSize).forEach(partition -> {
            MStream<Vector> input = StreamingContext.local().stream(partition);
            for (Layer layer : model.layers) {
               input = layer.preTrain(input, preTrainLR.get());
            }
            preTrainLR.set(preTrainLearningRate.get(preTrainLR.get(), iteration,
                                                    preTrainNumProcessed.addAndGet(preTrainBatchSize)));
         });
      }

      for (Layer layer : model.layers) {
         layerProcessor.accept(layer);
      }

      CostWeightTuple optimal = optimizer.optimize(theta,
                                                   dataset::asVectors,
                                                   new NeuralNetworkCostFunction(model, lossFunction),
                                                   TerminationCriteria.create().maxIterations(maxIterations)
                                                                      .historySize(3)
                                                                      .tolerance(tolerance),
                                                   learningRate,
                                                   weightUpdate,
                                                   verbose
                                                  );

      for (int i = 0, index = 0; i < numberOfLayers; i++) {
         if (model.layers[i].hasWeights()) {
            model.layers[i].setWeights(optimal.getComponents().get(index));
            index++;
         }
      }
      return model;
   }
}// END OF SequentialNetworkLearner
