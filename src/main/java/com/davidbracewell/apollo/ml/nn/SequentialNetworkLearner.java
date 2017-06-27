package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
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
   private int maxIterations = 100;
   @Getter
   @Setter
   private DifferentiableActivation outputActivation = new SigmoidActivation();


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

      final Layer OUTPUT_LAYER = model.layers[numberOfLayers - 1];

      for (int iteration = 0; iteration < maxIterations; iteration++) {
         double totalLoss = 0;
         for (int di = 0; di < data.size(); di++) {
            Vector m = data.get(di);
            Vector y = Vector.sZeros(nL).set((int) m.getLabelAsDouble(), 1);
            Vector[] layerOutput = new Vector[numberOfLayers];
            for (int i = 0; i < numberOfLayers; i++) {
               Layer layer = model.layers[i];
               layerOutput[i] = layer.forward(m);
               m = layerOutput[i];
            }

            Vector output = layerOutput[numberOfLayers - 1];
            totalLoss += lossFunction.loss(y, output);
            Vector delta = OUTPUT_LAYER.backward(output, y);

//            double factor = layerOutput[numberOfLayers - 1]
//                               .multiply(
//                                  model.layers[numberOfLayers - 1]
//                                     .backward(layerOutput[numberOfLayers - 1], y).T())
//                               .get(0, 0);

            for (int i = numberOfLayers - 2; i >= 0; i--) {
               Layer layer = model.layers[i];
               output = i > 0 ? layerOutput[i - 1] : m;
               totalLoss += lossFunction.loss(output, delta);
               delta = layer.backward(output, delta);
            }
         }
         System.out.println(totalLoss);
      }

      return model;
   }
}// END OF SequentialNetworkLearner
