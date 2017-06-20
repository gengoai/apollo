package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.Streams;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class BernoulliRBM {
   int nV;
   int nH;
   Matrix W;
   Vector vBias;
   Vector hBias;
   double learningRate = 0.1;

   public BernoulliRBM(int nV, int nH) {
      this.nV = nV;
      this.nH = nH;
      this.W = SparseMatrix.random(nV, nH);
      this.vBias = new SparseVector(nV);
      this.hBias = new SparseVector(nH);
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
      for (Vector vector : v) {
         Vector hid = rbm.runVisible(vector);
         Vector pred = rbm.runHidden(hid);
         System.err.println(vector + " : " + pred);
      }

      rbm.runVisible(new SparseVector(6).set(3, 1).set(4, 1));
   }

   public static double sigmoid(double x) {
      return 1. / (1. + Math.pow(Math.E, -x));
   }

   public int getNumHidden() {
      return nH;
   }

   public Vector runHidden(Vector v) {
      Vector activations = W.dot(v).add(vBias);
      Vector probs = activations.map(BernoulliRBM::sigmoid);
      return probs.map(d -> d > Math.random() ? 1 : 0);
   }

   public Vector runVisible(Vector v) {
      Vector activations = W.transpose().dot(v).add(hBias);
      Vector probs = activations.map(BernoulliRBM::sigmoid);
      return probs.map(d -> d > Math.random() ? 1 : 0);
   }

   private Matrix softmax(Matrix m) {
      double max = Streams.asStream(m.nonZeroIterator())
                          .mapToDouble(Matrix.Entry::getValue)
                          .max().orElse(0d);
      m = m.map(d -> d - max);
      double sum = m.sum();
      if (sum == 0) {
         return m;
      }
      return m.mapSelf(d -> d / sum);
   }

   public void train(List<Vector> data) {
      int numExamples = data.size();


      for (int epoch = 0; epoch < 5000; epoch++) {
         double error = 0;
         for (Vector datum : data) {
            Matrix posHiddenProbs = datum.toMatrix()
                                         .multiply(W)
                                         .addSelf(hBias.toMatrix())
                                         .mapSelf(BernoulliRBM::sigmoid);

            Matrix posHiddenStates = posHiddenProbs.map(d -> d > Math.random() ? 1 : 0);
            Matrix posAssociations = datum.transpose().multiply(posHiddenProbs);
            Matrix negVisibleProbs = posHiddenStates
                                        .multiply(W.transpose())
                                        .addSelf(vBias.toMatrix())
                                        .mapSelf(BernoulliRBM::sigmoid);
            Matrix negHiddenProbs = negVisibleProbs.multiply(W)
                                                   .mapSelf(BernoulliRBM::sigmoid);
            Matrix negAssociations = negVisibleProbs.transpose().multiply(negHiddenProbs);

            W.addSelf(posAssociations.subtract(negAssociations)
                                     .map(d -> learningRate * d / numExamples));

            error += datum.subtract(negVisibleProbs.row(0)).map(d -> FastMath.pow(d, 2)).sum();
         }
         System.err.println(String.format("Epoch %s: error is %s", epoch, error));
      }
      for (Vector vector : data) {
         Vector hid = runVisible(vector);
         Vector pred = runHidden(hid);
         System.err.println(vector + " : " + pred + " : " + hid);
      }
   }
}// END OF RBM2
