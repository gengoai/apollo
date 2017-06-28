package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.DecayLearningRate;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxActivation;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class SequentialNetworkLearner extends ClassifierLearner {
   @Getter
   @Setter
   private LossFunction lossFunction = new LogLoss();
   @Getter
   private List<Layer> layerConfiguration = new ArrayList<>();
   @Getter
   @Setter
   private int maxIterations = 500;
   @Getter
   @Setter
   private DifferentiableActivation outputActivation = new SoftmaxActivation();
   @Getter
   @Setter
   private LearningRate learningRate = new DecayLearningRate(0.1, 0.001);

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
      model.layers[0].setInputDimension(dataset.getFeatureEncoder().size());
      model.layers[numberOfLayers - 1] = new DenseLayer(outputActivation, dataset.getLabelEncoder().size());
      for (int i = 1; i < numberOfLayers; i++) {
         model.layers[i].connect(model.layers[i - 1]);
      }
      int nL = dataset.getLabelEncoder().size();
      List<Vector> data = dataset.asVectors().collect();

      double eta = getLearningRate().getInitialRate();
      int n = 0;
      for (int iteration = 1; iteration <= maxIterations; iteration++) {
         for (Vector input : data) {
            n++;
            eta = learningRate.get(eta, iteration, n);
            Vector y = Vector.sZeros(nL).set((int) input.getLabelAsDouble(), 1);
            Vector[] a = new Vector[numberOfLayers];
            Vector[] Fp = new Vector[numberOfLayers];
            for (int i = 0; i < numberOfLayers; i++) {
               Layer layer = model.layers[i];
               if (i == 0) {
                  a[i] = layer.forward(input);
                  Fp[i] = layer.calculateGradient(a[i]);
               } else {
                  a[i] = layer.forward(a[i - 1]);
                  if (i != numberOfLayers - 1) {
                     Fp[i] = layer.calculateGradient(a[i]);
                  }
               }
            }

            int nA = a.length;
            Vector[] d = new Vector[numberOfLayers];
            d[numberOfLayers - 1] = lossFunction.derivative(a[nA - 1], y)
                                                .multiplySelf(
                                                   model.layers[numberOfLayers - 1].calculateGradient(a[nA - 1]));
            for (int i = numberOfLayers - 2; i >= 0; i--) {
               d[i] = model.layers[i + 1].getWeights().T()
                                         .multiply(d[i + 1].transpose())
                                         .column(0)
                                         .multiply(Fp[i]);
            }
            for (int i = 0; i < numberOfLayers - 1; i++) {
               Matrix Wi = model.layers[i].getWeights();
               Vector ai = (i == 0) ? input : a[i - 1];
               Wi.subtractSelf(d[i].transpose().multiply(ai.toMatrix()).scaleSelf(eta));
            }
         }
      }

      return model;
   }
}// END OF SequentialNetworkLearner
