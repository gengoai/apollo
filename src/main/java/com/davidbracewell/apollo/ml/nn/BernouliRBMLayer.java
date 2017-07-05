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
      this.inputSize = previousLayer.getInputSize();
      this.weights = new Weights(DenseMatrix.zeroes(outputSize, inputSize),
                                 DenseVector.zeros(outputSize),
                                 outputSize <= 2
      );
      this.visibleBias = Vector.sZeros(inputSize);
   }

   @Override
   public Vector forward(Vector input) {
      return SigmoidActivation.INSTANCE.apply(this.weights.dot(input.add(visibleBias)));
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
   public MStream<Vector> pretrain(MStream<Vector> previousOutput) {
      List<Vector> vectors = previousOutput.collect();
      List<Vector> out = vectors.stream().map(v -> forward(v).map(d -> d >= Math.random() ? 1.0 : 0.0))
                                .collect(Collectors.toList());


      Matrix m = new DenseMatrix(vectors.size(), inputSize);
      for (int i = 0; i < vectors.size(); i++) {
         m.setRow(i, vectors.get(i));
      }

      Matrix positiveHiddenProbabilities = m.multiply(weights.getTheta().T())
                                            .addRowSelf(weights.getBias())
                                            .mapSelf(SigmoidActivation.INSTANCE::apply);
      Matrix positiveHiddenStates = positiveHiddenProbabilities.map(d -> d >= Math.random() ? 1 : 0);
      Matrix positiveAssociations = m.T().multiply(positiveHiddenProbabilities);

//      negative CD phase
      Matrix negativeVisibleProbabilities = positiveHiddenStates.multiply(weights.getTheta())
                                                                .addRowSelf(visibleBias)
                                                                .mapSelf(SigmoidActivation.INSTANCE::apply);
      negativeVisibleProbabilities.setColumn(0, DenseVector.ones(negativeVisibleProbabilities.numberOfRows()));

      Matrix negativeHiddenProbabilities = negativeVisibleProbabilities
                                              .multiply(weights.getTheta().T())
                                              .mapSelf(SigmoidActivation.INSTANCE::apply);
      Matrix negativeAssociations = negativeVisibleProbabilities.T().multiply(negativeHiddenProbabilities);

      Matrix gradient = positiveAssociations.subtract(negativeAssociations)
                                            .map(d -> d / vectors.size()).T();
      weights.getTheta().addSelf(gradient.scaleSelf(0.01));

      Vector rowSum = Vector.dZeros(gradient.numberOfColumns());


      gradient.rowIterator().forEachRemaining(rowSum::addSelf);

      Vector colSum = Vector.dZeros(gradient.numberOfRows());
      gradient.columnIterator().forEachRemaining(colSum::addSelf);

      weights.getBias().addSelf(colSum);
      visibleBias.addSelf(rowSum);
      return StreamingContext.local().stream(out);
   }
}// END OF RBMLayer
