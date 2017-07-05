package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;

import java.util.List;
import java.util.stream.Collectors;

/**
 * The type Bernouli rbm layer.
 *
 * @author David B. Bracewell
 */
public class BernouliRBMLayer implements Layer {
   private static final long serialVersionUID = 1L;
   private final int outputSize;
   private Weights weights;
   private Vector visibleBias;
   private int inputSize;

   /**
    * Instantiates a new Bernouli rbm layer.
    *
    * @param outputSize the output size
    */
   public BernouliRBMLayer(int outputSize) {
      this.outputSize = outputSize;
   }

   @Override
   public Vector backward(Vector output, Vector delta) {
      return delta
                .multiply(SigmoidActivation.INSTANCE.valueGradient(output))
                .toMatrix()
                .multiply(weights.getTheta())
                .row(0);
   }

   @Override
   public void connect(Layer previousLayer) {
      this.inputSize = previousLayer.getOutputSize();
      this.weights = new Weights(DenseMatrix.zeroes(outputSize, inputSize),
                                 DenseVector.zeros(outputSize),
                                 outputSize <= 2
      );
      this.visibleBias = Vector.sZeros(inputSize);
   }

   @Override
   public Vector forward(Vector input) {
      return SigmoidActivation.INSTANCE.apply(this.weights.getTheta().dot(input.add(visibleBias)).column(0));
   }

   @Override
   public int getInputSize() {
      return inputSize;
   }

   @Override
   public int getOutputSize() {
      return outputSize;
   }

   @Override
   public Weights getWeights() {
      return weights;
   }

   @Override
   public void setWeights(Weights weights) {
      this.weights = weights;
   }

   @Override
   public boolean hasWeights() {
      return true;
   }

   @Override
   public MStream<Vector> preTrain(MStream<Vector> previousOutput, double learningRate) {
      List<Vector> vectors = previousOutput.collect();
      List<Vector> out = vectors.stream().map(v -> forward(v).map(d -> d >= Math.random() ? 1.0 : 0.0))
                                .collect(Collectors.toList());


      Matrix m = new DenseMatrix(vectors.size(), inputSize + 1);
      for (int i = 0; i < vectors.size(); i++) {
         m.setRow(i, vectors.get(i).insert(0, 1));
      }

      Matrix theta = weights.getTheta();
      Matrix W = DenseMatrix.zeroes(theta.numberOfRows() + 1, theta.numberOfColumns() + 1);
      for (int r = 1; r <= theta.numberOfRows(); r++) {
         W.setRow(r, theta.row(r - 1).insert(0, 0));
      }

      Matrix positiveHiddenProbabilities = m.multiply(W.T()).mapSelf(SigmoidActivation.INSTANCE::apply);
      Matrix positiveHiddenStates = positiveHiddenProbabilities.map(d -> d > Math.random() ? 1 : 0);
      Matrix positiveAssociations = m.T().multiply(positiveHiddenProbabilities);

//      negative CD phase
      //negative CD phase
      Matrix negativeVisibleProbabilities = positiveHiddenStates.multiply(W)
                                                                .mapSelf(SigmoidActivation.INSTANCE::apply);
      negativeVisibleProbabilities.setColumn(0, DenseVector.ones(negativeVisibleProbabilities.numberOfRows()));

      Matrix negativeHiddenProbabilities = negativeVisibleProbabilities.multiply(W.T())
                                                                       .mapSelf(SigmoidActivation.INSTANCE::apply);
      Matrix negativeAssociations = negativeVisibleProbabilities.T().multiply(negativeHiddenProbabilities);

      W.addSelf(positiveAssociations.subtract(negativeAssociations)
                                    .map(d -> learningRate * d / vectors.size()).T());
      weights.setTheta(W.slice(1, 1));
      weights.setBias(W.column(0).slice(1).copy());
      visibleBias = W.row(0).slice(1).copy();
      return StreamingContext.local().stream(out);
   }
}// END OF RBMLayer
