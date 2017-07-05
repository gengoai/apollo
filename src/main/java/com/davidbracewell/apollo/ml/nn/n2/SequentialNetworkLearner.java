package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.update.DeltaRule;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
@Builder
@AllArgsConstructor
public class SequentialNetworkLearner extends ClassifierLearner {
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
   private DifferentiableActivation outputActivation = new SigmoidActivation();
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
   private int batchSize = -1;
   @Getter
   @Setter
   @Builder.Default
   private double tolerance = 1e-4;
   @Getter
   @Setter
   @Builder.Default
   private boolean verbose = false;

   public SequentialNetworkLearner() {

   }

   public SequentialNetworkLearner add(Layer layer) {
      layers.add(layer);
      return this;
   }

   private Layer[] makeLayers(int numberOfFeatures, int numberOfLabels) {
      List<Layer> layerList = new ArrayList<>(layers);
      layerList.add(new OutputLayer(outputActivation, numberOfLabels));
      InputLayer inputLayer = new InputLayer(numberOfFeatures);
      layerList.get(0).connect(inputLayer);
      for (int i = 1; i < layerList.size(); i++) {
         layerList.get(i).connect(layerList.get(i - 1));
      }
      return layerList.toArray(new Layer[layerList.size()]);
   }

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      SequentialNetwork model = new SequentialNetwork(this);

      final int numberOfLayers = layers.size() + 1;
      model.layers = makeLayers(dataset.getFeatureEncoder().size(), dataset.getLabelEncoder().size());


      int numWeights = 0;
      for (int i = 0; i < numberOfLayers; i++) {
         if (model.layers[i].isOptimizable()) {
            numWeights++;
         }
      }

      int[][] shapes = new int[numWeights][2];

      for (int i = 0, index = 0; i < numberOfLayers; i++) {
         if (model.layers[i].isOptimizable()) {
            shapes[index][0] = model.layers[i].getOutputSize();
            shapes[index][1] = (i == 0) ? dataset.getFeatureEncoder().size() : model.layers[i - 1].getOutputSize();
            index++;
         }
      }
      WeightComponent theta = new WeightComponent(shapes, WeightInitializer.ZEROES);
      for (int i = 0, index = 0; i < numberOfLayers; i++) {
         if (model.layers[i].isOptimizable()) {
            theta.set(index, model.layers[i].getWeights());
            index++;
         }
      }

      Optimizer optimizer;
      if (batchSize > 0) {
         optimizer = new BatchOptimizer(new SGD(), batchSize);
      } else {
         optimizer = new SGD();
      }
      CostWeightTuple optimal = optimizer.optimize(theta,
                                                   dataset::asVectors,
                                                   new SequentialNetworkCostFunction(model, lossFunction),
                                                   TerminationCriteria.create().maxIterations(maxIterations)
                                                                      .historySize(3)
                                                                      .tolerance(tolerance),
                                                   learningRate,
                                                   weightUpdate,
                                                   verbose
                                                  );

      for (int i = 0, index = 0; i < numberOfLayers; i++) {
         if (model.layers[i].isOptimizable()) {
            model.layers[i].setWeights(optimal.getComponents().get(index));
            index++;
         }
      }
      return model;
   }
}// END OF SequentialNetworkLearner
