package com.davidbracewell.apollo.ml.nn;

/**
 * @author David B. Bracewell
 */
public class FeedForwardNetworkLearner {
   //extends ClassifierLearner {
//   @Getter
//   private List<Layer> layers = new ArrayList<>();
//   @Getter
//   private Layer outputLayer = new HiddenLayer(SigmoidActivation.INSTANCE);
//   @Getter
//   @Setter
//   private int maxIterations = 100;
//
//   public void addHiddenLayer(@NonNull Layer hiddenLayer) {
//      this.layers.add(hiddenLayer);
//   }
//
//   @Override
//   public void resetLearnerParameters() {
//      layers.forEach(Layer::reset);
//   }
//
//   public void setHiddenLayers(@NonNull List<Layer> hiddenLayers) {
//      this.layers = hiddenLayers;
//   }
//
//   public void setOutputLayer(@NonNull Layer outputLayer) {
//      this.outputLayer = outputLayer;
//   }
//
//   @Override
//   protected Classifier trainImpl(Dataset<Instance> dataset) {
//      int nIn = dataset.getFeatureEncoder().size();
//      for (Layer layer : layers) {
//         layer.init(nIn);
//         nIn = layer.getOutputSize();
//      }
//      outputLayer.init(nIn, dataset.getLabelEncoder().size());
//      Matrix data = dataset.toDenseMatrix();
//      int nL = dataset.getLabelEncoder().size();
//      Matrix y = DenseMatrix.zeroes(dataset.size(), nL);
//      int index = 0;
//      for (Instance instance : dataset) {
//         y.setRow(index, SparseVector.zeros(nL).set((int) dataset.getLabelEncoder().encode(instance.getLabel()), 1));
//         index++;
//      }
//      List<Layer> allLayers = Lists.union(layers, Collections.singletonList(outputLayer));
//
//      for (int iteration = 0; iteration < maxIterations; iteration++) {
//         Matrix m = data;
//
//         Matrix[] steps = new Matrix[layers.size()];
//         for (int i = 0; i < layers.size(); i++) {
//            Layer layer = layers.get(i);
//            steps[i] = layer.forward(m);
//            m = steps[i];
//         }
//
//         Matrix totalError = DenseMatrix.zeroes(data.numberOfRows(), nL);
//         Matrix error = y.subtract(steps[layers.size() - 1]).mapSelf(d -> 0.5 * (d * d));
//
//         for (int i = layers.size() - 1; i >= 0; i--) {
//            Layer layer = layers.get(i);
//
//         }
//
//      }
//
//      FeedForwardNetwork ffn = new FeedForwardNetwork(this);
//      ffn.layers = allLayers.toArray(new Layer[1]);
//      return ffn;
//   }
}// END OF FeedForwardNetworkLearner
