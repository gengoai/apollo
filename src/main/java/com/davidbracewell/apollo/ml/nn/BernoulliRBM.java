package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.*;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class BernoulliRBM implements Layer {
   int nV;
   int nH;
   Matrix W;
   double learningRate = 0.1;

   public BernoulliRBM(int nV, int nH) {
      this.nV = nV;
      this.nH = nH;
      this.W = SparseMatrix.random(nV + 1, nH + 1);
   }

   public static void main(String[] args) {
      BernoulliRBM rbm = new BernoulliRBM(6, 2);
      List<Vector> v = new ArrayList<>();
      v.add(new SparseVector(6).set(0, 1).set(1, 1).set(2, 1));
      v.add(new SparseVector(6).set(0, 1).set(2, 1));
      v.add(new SparseVector(6).set(0, 1).set(1, 1).set(2, 1));
      v.add(new SparseVector(6).set(2, 1).set(3, 1).set(4, 1));
      v.add(new SparseVector(6).set(2, 1).set(4, 1));
      v.add(new SparseVector(6).set(2, 1).set(3, 1).set(4, 1));
      rbm.train(v);
      System.out.println(rbm.W.row(0));
      System.out.println(rbm.W.column(0));
      for (Vector vector : v) {
         Vector hid = rbm.runVisible(vector);
         Vector pred = rbm.runHidden(hid);
         System.err.println(vector + " : " + pred);
      }

      rbm.runVisible(new SparseVector(6).set(3, 1).set(4, 1));
   }

   @Override
   public Matrix forward(Matrix input) {
      return new SparseMatrix(1, runVisible(input.row(0)));
   }

   @Override
   public int getInputSize() {
      return nV;
   }

   public int getNumHidden() {
      return nH;
   }

   @Override
   public int getOutputSize() {
      return nV;
   }

   @Override
   public void init(int nIn) {
      this.nV = nIn;
      this.W = SparseMatrix.random(nV + 1, nH + 1);
   }

   @Override
   public void init(int nIn, int nOut) {
      this.nH = nOut;
      init(nIn);
   }

   @Override
   public void reset() {
      this.W = SparseMatrix.random(nV + 1, nH + 1);
   }

   public Vector runHidden(Vector v) {
      Matrix m = new DenseMatrix(1, nH + 1);
      m.setRow(0, v.insert(0, 1));
      Matrix visibleStates = m.multiply(W.T())
                              .mapSelf(d -> SigmoidActivation.INSTANCE.apply(d) > Math.random() ? 1 : 0);
      return visibleStates.row(0).slice(1, nV + 1);
   }

   public Vector runVisible(Vector v) {
      Matrix m = new DenseMatrix(1, nV + 1);
      m.setRow(0, v.insert(0, 1));
      Matrix hiddenStates = m.multiply(W)
                             .mapSelf(d -> SigmoidActivation.INSTANCE.apply(d) > Math.random() ? 1 : 0);
      return hiddenStates.row(0).slice(1, nH + 1);
   }

   public void train(List<Vector> data) {
      int numExamples = data.size();

      Matrix m = new DenseMatrix(data.size(), nV + 1);
      for (int i = 0; i < data.size(); i++) {
         m.setRow(i, data.get(i).insert(0, 1));
      }

      final Random rnd = new Random();
      for (int epoch = 0; epoch < 5000; epoch++) {
         double error = 0;
         //positive CD phase
         Matrix positiveHiddenProbabilities = m.multiply(W).mapSelf(SigmoidActivation.INSTANCE::apply);
         Matrix positiveHiddenStates = positiveHiddenProbabilities.map(d -> d > rnd.nextGaussian() ? 1 : 0);
         Matrix positiveAssociations = m.T().multiply(positiveHiddenProbabilities);

         //negative CD phase
         Matrix negativeVisibleProbabilities = positiveHiddenStates.multiply(W.T())
                                                                   .mapSelf(SigmoidActivation.INSTANCE::apply);
         negativeVisibleProbabilities.setColumn(0, DenseVector.ones(negativeVisibleProbabilities.numberOfRows()));

         Matrix negativeHiddenProbabilities = negativeVisibleProbabilities.multiply(W)
                                                                          .mapSelf(SigmoidActivation.INSTANCE::apply);
         Matrix negativeAssociations = negativeVisibleProbabilities.T().multiply(negativeHiddenProbabilities);

         W.addSelf(positiveAssociations.subtract(negativeAssociations)
                                       .map(d -> learningRate * d / numExamples));
         error += m.subtractRow(negativeVisibleProbabilities.row(0)).map(d -> FastMath.pow(d, 2)).sum();
         if (epoch % 100 == 0 || (epoch + 1) == 5000) {
            System.err.println(String.format("Epoch %s: error is %s", epoch, error));
         }
      }
   }
}// END OF RBM2
