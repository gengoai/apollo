package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.BottouLearningRate;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.apollo.optimization.update.DeltaRule;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Collections;
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

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      SequentialNetwork model = new SequentialNetwork(this);

      final int numberOfLayers = layerConfiguration.size() + 1;

      model.layers = new Layer[numberOfLayers];
      System.arraycopy(layerConfiguration.toArray(new Layer[1]), 0, model.layers, 0, numberOfLayers - 1);
      InputLayer inputLayer = new InputLayer(dataset.getFeatureEncoder().size());
      model.layers[0].connect(inputLayer);
      model.layers[numberOfLayers - 1] = new OutputLayer(outputActivation, dataset.getLabelEncoder().size());
      for (int i = 1; i < numberOfLayers; i++) {
         model.layers[i].connect(model.layers[i - 1]);
      }
      int nL = dataset.getLabelEncoder().size();
      List<Vector> data = dataset.asVectors().collect();

      double eta = getLearningRate().getInitialRate();
      int n = 0;
      for (int iteration = 1; iteration <= maxIterations; iteration++) {
         Collections.shuffle(data);
         double totalError = 0;
         for (Vector input : data) {
            n++;
            eta = learningRate.get(eta, iteration, n);
            Vector y = Vector.sZeros(nL).set((int) input.getLabelAsDouble(), 1);

            Vector[] activations = new Vector[model.layers.length];
            for (int i = 0; i < model.layers.length; i++) {
               if (i == 0) {
                  activations[i] = model.layers[i].forward(input);
               } else {
                  activations[i] = model.layers[i].forward(activations[i - 1]);
               }
            }
            Vector predicted = activations[activations.length - 1];

            totalError += lossFunction.loss(predicted, y);
//            Vector delta = lossFunction.derivative(predicted, y);
//            for (int i = model.layers.length - 1; i >= 0; i--) {
//               Vector a = i == 0 ? input : activations[i - 1];
//               delta = model.layers[i].backward(a, activations[i], delta, weightUpdate, eta);
//            }
         }

         if (iteration % 50 == 0 || iteration == maxIterations - 1) {
            System.out.println("iteration=" + iteration + ", totalError=" + totalError);
         }

      }

      return model;
   }
}// END OF SequentialNetworkLearner
