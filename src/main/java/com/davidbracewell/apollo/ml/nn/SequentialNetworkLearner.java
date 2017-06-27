package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.loss.LossFunction;
import com.davidbracewell.collection.list.Lists;
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
   @Setter
   private LossFunction lossFunction = new LogLoss();
   @Getter
   private List<Layer> layers = new ArrayList<>();
   @Getter
   @Setter
   private int maxIterations = 100;
   @Getter
   @Setter
   private ActivationLayer outputLayer = new ActivationLayer(new SigmoidActivation());


   public SequentialNetworkLearner add(Layer layer) {
      layers.add(layer);
      return this;
   }

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      SequentialNetwork model = new SequentialNetwork(this);
      model.layers = new Layer[layers.size() + 2];
      System.arraycopy(layers.toArray(new Layer[1]), 0, model.layers, 0, layers.size());
      model.layers[0].setInputDimension(dataset.getFeatureEncoder().size());
      for (int i = 1; i < layers.size(); i++) {
         model.layers[i].connect(model.layers[i - 1]);
      }
      model.layers[layers.size()] = new DenseLayer(dataset.getLabelEncoder().size());
      model.layers[layers.size()].connect(model.layers[layers.size() - 1]);
      model.layers[layers.size() + 1] = outputLayer;
      model.layers[layers.size() + 1].connect(model.layers[layers.size()]);


      Matrix data = dataset.toDenseMatrix();
      int nL = dataset.getLabelEncoder().size();
      Matrix y = DenseMatrix.zeroes(dataset.size(), nL);
      int index = 0;
      for (Instance instance : dataset) {
         y.setRow(index, SparseVector.zeros(nL).set((int) dataset.getLabelEncoder().encode(instance.getLabel()), 1));
         index++;
      }
      List<Layer> allLayers = Lists.union(layers, Collections.singletonList(outputLayer));

      for (int iteration = 0; iteration < maxIterations; iteration++) {
         Matrix m = data;

         Matrix[] steps = new Matrix[layers.size()];
         for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            steps[i] = layer.forward(m);
            m = steps[i];
         }

         Matrix totalError = DenseMatrix.zeroes(data.numberOfRows(), nL);
         Matrix error = y.subtract(steps[layers.size() - 1]).mapSelf(d -> 0.5 * (d * d));

         for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);

         }

      }

      SequentialNetwork ffn = new SequentialNetwork(this);
      ffn.layers = allLayers.toArray(new Layer[1]);
      return ffn;
   }
}// END OF SequentialNetworkLearner
