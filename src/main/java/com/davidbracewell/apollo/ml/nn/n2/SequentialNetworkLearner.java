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
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class SequentialNetworkLearner extends ClassifierLearner {
   @Getter
   private List<Layer> layerConfiguration = new ArrayList<>();
   @Getter
   @Setter
   private int maxIterations = 200;
   @Getter
   @Setter
   private DifferentiableActivation outputActivation = new SigmoidActivation();
   @Getter
   @Setter
   private LearningRate learningRate = new BottouLearningRate(0.1, 0.001);
   @Getter
   @Setter
   private LossFunction lossFunction = new LogLoss();
   @Getter
   @Setter
   private WeightUpdate weightUpdate = new DeltaRule();

   public SequentialNetworkLearner add(Layer layer) {
      layerConfiguration.add(layer);
      return this;
   }

   private Layer[] makeLayers(int numberOfFeatures, int numberOfLabels) {
      List<Layer> layerList = new ArrayList<>(layerConfiguration);
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

      final int numberOfLayers = layerConfiguration.size() + 1;
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

      Optimizer optimizer = new SGD();
      CostWeightTuple optimal = optimizer.optimize(theta,
                                                   dataset::asVectors,
                                                   new SequentialNetworkCostFunction(model, lossFunction),
                                                   TerminationCriteria.create().maxIterations(maxIterations)
                                                                      .historySize(3)
                                                                      .tolerance(1e-4),
                                                   learningRate,
                                                   weightUpdate,
                                                   true
                                                  );

      for (int i = 0, index = 0; i < numberOfLayers; i++) {
         if (model.layers[i].isOptimizable()) {
            model.layers[i].setWeights(optimal.getComponents().get(index));
            index++;
         }
      }

//      double eta = getLearningRate().getInitialRate();
//      int n = 0;
//      for (int iteration = 1; iteration <= maxIterations; iteration++) {
//         Collections.shuffle(data);
//         double totalError = 0;
//         for (Vector input : data) {
//            n++;
//            eta = learningRate.get(eta, iteration, n);
//            Vector y = Vector.sZeros(nL).set((int) input.getLabelAsDouble(), 1);
//
//            Vector[] activations = new Vector[model.layers.length];
//            for (int i = 0; i < model.layers.length; i++) {
//               if (i == 0) {
//                  activations[i] = model.layers[i].forward(input);
//               } else {
//                  activations[i] = model.layers[i].forward(activations[i - 1]);
//               }
//            }
//            Vector predicted = activations[activations.length - 1];
//
//            totalError += lossFunction.loss(predicted, y);
//            Gradient[] deltas = new Gradient[model.layers.length];
//            deltas[model.layers.length - 1] = lossFunction.derivative(predicted, y);
//            for (int i = model.layers.length - 1; i >= 0; i--) {
//               Vector a = i == 0 ? input : activations[i - 1];
//               deltas[i] = model.layers[i].backward(a, activations[i], deltas[i + 1]);
//            }
//         }
//
//         if (iteration % 50 == 0 || iteration == maxIterations - 1) {
//            System.out.println("iteration=" + iteration + ", totalError=" + totalError);
//         }
//
//      }

      return model;
   }
}// END OF SequentialNetworkLearner
