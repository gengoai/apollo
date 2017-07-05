package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.*;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

/**
 * The type Bernoulli rbm.
 *
 * @author David B. Bracewell
 */
public class BernoulliRBM implements Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The N v.
    */
   int nV;
   /**
    * The W.
    */
   Matrix W;
   /**
    * The Learning rate.
    */
   double learningRate = 0.1;
   /**
    * The N h.
    */
   private int nH;

   /**
    * Instantiates a new Bernoulli rbm.
    *
    * @param nV the n v
    * @param nH the n h
    */
   public BernoulliRBM(int nV, int nH) {
      this.nV = nV;
      this.nH = nH;
      this.W = SparseMatrix.random(nV + 1, nH + 1);
   }

   /**
    * Gets num hidden.
    *
    * @return the num hidden
    */
   public int getNumHidden() {
      return nH;
   }

   /**
    * Init.
    *
    * @param nIn the n in
    */
   public void init(int nIn) {
      this.nV = nIn;
      this.W = SparseMatrix.random(nV + 1, nH + 1);
   }

   /**
    * Init.
    *
    * @param nIn  the n in
    * @param nOut the n out
    */
   public void init(int nIn, int nOut) {
      this.nH = nOut;
      init(nIn);
   }

   /**
    * Reset.
    */
   public void reset() {
      this.W = SparseMatrix.random(nV + 1, nH + 1);
   }

   /**
    * Run hidden vector.
    *
    * @param v the v
    * @return the vector
    */
   public Vector runHidden(Vector v) {
      return runHiddenProbs(v).mapSelf(d -> d > Math.random() ? 1 : 0);
   }

   /**
    * Run hidden probs vector.
    *
    * @param v the v
    * @return the vector
    */
   public Vector runHiddenProbs(Vector v) {
      Matrix m = new DenseMatrix(1, nH + 1);
      m.setRow(0, v.insert(0, 1));
      return m.multiply(W.T()).mapSelf(SigmoidActivation.INSTANCE::apply).row(0).slice(1, nV + 1);
   }

   /**
    * Run visible vector.
    *
    * @param v the v
    * @return the vector
    */
   public Vector runVisible(Vector v) {
      return runVisibleProbs(v).mapSelf(d -> d > Math.random() ? 1 : 0);
   }

   /**
    * Run visible probs vector.
    *
    * @param v the v
    * @return the vector
    */
   public Vector runVisibleProbs(Vector v) {
      Matrix m = new DenseMatrix(1, nV + 1);
      m.setRow(0, v.insert(0, 1));
      return m.multiply(W).mapSelf(SigmoidActivation.INSTANCE::apply).row(0).slice(1, nH + 1);
   }

   /**
    * Train.
    *
    * @param data the data
    */
   public void train(List<Vector> data, int maxIterations) {
      int numExamples = data.size();

      Matrix m = new DenseMatrix(data.size(), nV + 1);
      for (int i = 0; i < data.size(); i++) {
         System.out.println(m.shape() + " : " + data.get(i).dimension());
         m.setRow(i, data.get(i).insert(0, 1));
      }

      final Random rnd = new Random();
      for (int epoch = 0; epoch < maxIterations; epoch++) {
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
