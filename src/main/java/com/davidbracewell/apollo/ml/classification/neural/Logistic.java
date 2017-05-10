package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierEvaluation;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;

import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class Logistic {
   final int outputSize;
   final int inputSize;
   final Vector[] weights;
   final Vector bias;


   public Logistic(int inputSize, int outputSize) {
      this.outputSize = outputSize;
      this.inputSize = inputSize;
      this.bias = SparseVector.zeros(outputSize);
      this.weights = new Vector[outputSize];
      for (int i = 0; i < outputSize; i++) {
         this.weights[i] = SparseVector.zeros(inputSize);
      }
   }

   public static void main(String[] args) {
      Resource url = Resources.from(
         "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/iris_binned.csv");
      DenseCSVDataSource dataSource = new DenseCSVDataSource(url, true);
      dataSource.setLabelName("Species");
      Dataset<Instance> dataset = Dataset.classification()
                                         .source(dataSource)
                                         .shuffle(new Random(1234));

      dataset.encode();
      HiddenLayer hidden = new HiddenLayer(dataset.getFeatureEncoder().size(),
                                           10,
                                           v -> Math.max(0, v),
                                           v -> v > 0 ? 1 : 0
      );
      Logistic logistic = new Logistic(10, dataset.getLabelEncoder().size());
      MStream<FeatureVector> v = dataset.asFeatureVectors().cache();
      AtomicDouble lr = new AtomicDouble(1);
      for (int i = 0; i < 500; i++) {
         v.shuffle().partition(10)
          .forEach(it -> {
             FeatureVector[] vv = Streams.asStream(it).toArray(FeatureVector[]::new);
             Vector[] vf = new Vector[vv.length];
             for (int n = 0; n < vv.length; n++) {
                vf[n] = hidden.forward(vv[n]);
             }
             Vector[] dy = logistic.train(vf, vf.length, lr.get());
             hidden.backward(vv, vf, dy, vv.length, lr.get());
          });
         lr.set(lr.get() * .95);
      }

      ClassifierEvaluation evaluation = new ClassifierEvaluation();
      v.forEach(vp -> {
         Vector p = logistic.predict(hidden.forward(vp));
         int i = 0;
         double max = 0;
         for (Vector.Entry e : p) {
            if (e.getValue() > max) {
               max = e.getValue();
               i = e.getIndex();
            }
         }
         evaluation.entry(vp.getDecodedLabel(), dataset.getLabelEncoder().decode((double) i).toString());
      });

      evaluation.output(System.out);


   }

   public Vector predict(Vector input) {
      Vector out = new DenseVector(weights.length);
      for (int j = 0; j < weights.length; j++) {
         out.set(j, input.dot(weights[j]) + bias.get(j));
      }
      double max = out.max();
      out.mapSelf(v -> Math.exp(v - max));
      double sum = out.sum();
      out.mapDivideSelf(sum);
      return out;
   }

   public Vector[] train(Vector[] X, int minibatchSize, double learningRate) {
      Vector[] gradW = new Vector[outputSize];
      for (int i = 0; i < outputSize; i++) {
         gradW[i] = SparseVector.zeros(inputSize);
      }
      Vector gradB = SparseVector.zeros(outputSize);
      Vector[] dY = new Vector[minibatchSize];
      for (int n = 0; n < minibatchSize; n++) {
         dY[n] = SparseVector.zeros(outputSize);
         Vector pred_y = predict(X[n]);
         Vector y = SparseVector.zeros(outputSize);
         y.set(((Double) X[n].getLabel()).intValue(), 1);

         for (int j = 0; j < outputSize; j++) {
            dY[n].set(j, pred_y.get(j) - y.get(j));
            for (int i = 0; i < inputSize; i++) {
               gradW[j].increment(i, dY[n].get(j) * X[n].get(i));
            }

            gradB.increment(j, dY[n].get(j));
         }

      }


      for (int j = 0; j < outputSize; j++) {
         for (int i = 0; i < inputSize; i++) {
            weights[j].decrement(i, learningRate * gradW[j].get(j) / minibatchSize);
         }
         bias.decrement(j, learningRate * gradB.get(j) / minibatchSize);
      }


      return dY;
   }


}// END OF Logistic
