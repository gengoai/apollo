package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.apollo.optimization.BottouLearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.alt.LogLoss;
import com.davidbracewell.apollo.optimization.alt.again.*;
import com.davidbracewell.guava.common.base.Stopwatch;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;

import java.util.Random;

import static com.davidbracewell.apollo.ml.classification.ClassifierEvaluation.crossValidation;

/**
 * Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
 *
 * @author David B. Bracewell
 */
public class LrLearner extends BinaryClassifierLearner {

   public static void main(String[] args) {
      Resource url = Resources.from(
         "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/iris_headers.csv");


      DenseCSVDataSource dataSource = new DenseCSVDataSource(url, true);
      dataSource.setLabelName("Species");
      Dataset<Instance> dataset = Dataset.classification()
                                         .source(dataSource)
                                         .shuffle(new Random(1234));


      Stopwatch sw = Stopwatch.createStarted();
      crossValidation(dataset, () -> new CCLearner(), 10).output(System.out);
      sw.stop();
      System.out.println(sw);

//      crossValidation(dataset, () -> new BGD().oneVsRest(), 10).output(System.out);
//      crossValidation(dataset, SoftmaxLearner::new, 10)
//         .output(System.out);
   }

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected LrClassifier trainForLabel(Dataset<Instance> dataset, double trueLabel) {
      LrClassifier model = new LrClassifier(this);

      model.bias = 0;
      model.weights = new SparseVector(dataset.getEncoderPair().numberOfFeatures());
      final int N = 1;

      Vector q = new SparseVector(dataset.getEncoderPair().numberOfFeatures() + 1);
      double eta0 = 1;
      double C = 1;
      double u = 0;
      double tolerance = 0.000001;

      double previous = 0;

      for (int iteration = 1; iteration <= 200; iteration++) {
         double eta = (eta0 + 1.0) / (1.0 + ((double) iteration / (double) N));
         u = u + eta * C / N;
         double sumLogLikelihood = 0;
         for (Instance instance : dataset.shuffle()) {
            double y = dataset.getEncoderPair().encodeLabel(instance.getLabel()) == trueLabel ? 1 : 0;
            double yHat = model.classify(instance).distribution()[(int) y];
            double gradient = y - yHat;
            sumLogLikelihood += y * Math.log(yHat + 1e-24) + (1 - y) * Math.log(1 - yHat + 1e-24);
            if (gradient != 0) {
               for (Feature feature : instance) {
                  int fid = (int) dataset.getEncoderPair().encodeFeature(feature.getName());
                  model.weights.increment(fid, gradient * eta * feature.getValue());

                  double z = model.weights.get(fid);
                  double wi = model.weights.get(fid);
                  if (model.weights.get(fid) > 0) {
                     double v = Math.max(0, wi - (u + q.get(fid)));
                     model.weights.set(fid, v);
                  } else if (model.weights.get(fid) < 0) {
                     double v = Math.min(0, wi + (u - q.get(fid)));
                     model.weights.set(fid, v);
                  }
                  q.increment(fid, model.weights.get(fid) - z);
               }

               model.bias *= gradient * eta;
            }
         }
         sumLogLikelihood = -sumLogLikelihood;

         if (Math.abs(sumLogLikelihood - previous) <= tolerance) {
            break;
         }

         previous = sumLogLikelihood;
      }

      model.weights.mapDivideSelf(model.weights.l1Norm());
      return model;
   }

   public static class CCLearner extends ClassifierLearner {

      @Override
      protected void resetLearnerParameters() {

      }

      @Override
      protected Classifier trainImpl(Dataset<Instance> dataset) {
         int nL = dataset.getLabelEncoder().size();
         int nF = dataset.getFeatureEncoder().size();

         MStream<Vector> vectors = dataset.asVectors().cache();
         WeightMatrix weights = new WeightMatrix(nL, nF);
         SGD sgd = new SGD();
         CostWeightTuple cwt = sgd.optimize(weights,
                                            () -> vectors,
                                            new GradientDescentCostFunction(new LogLoss(), Activation.SOFTMAX),
                                            TerminationCriteria.create()
                                                               .maxIterations(200)
                                                               .historySize(3)
                                                               .tolerance(1e-4),
                                            new BottouLearningRate(1.0, 0.9),
                                            new DeltaRule(),
                                            5
                                           );

//
//         CrossEntropyLoss cel = new CrossEntropyLoss();
//         double lr = 0.1;
//         TerminationCriteria tc = TerminationCriteria.create()
//                                                     .maxIterations(200)
//                                                     .historySize(3)
//                                                     .tolerance(1e-4);
//         for (int iteration = 0; iteration < tc.maxIterations(); iteration++) {
//            double loss = 0;
//            for (Vector v : data) {
//               Vector predicted =
//               Vector predicted = Vector.sZeros(nL);
//               for (int i = 0; i < weights.length; i++) {
//                  predicted.set(i, weights[i].dot(v));
//               }
//               predicted = Activation.SOFTMAX.apply(predicted);
//               Vector y = v.getLabelVector(nL);
//               loss += cel.loss(predicted, y);
//               Vector gradient = predicted.subtract(y);
//               for (int i = 0; i < weights.length; i++) {
//                  weights[i].update(Gradient.of(v.mapMultiply(gradient.get(i) * lr), gradient.get(i) * lr));
//               }
//            }
////            System.out.println("iteration=" + iteration + ", loss=" + loss);
//            if (tc.check(loss)) {
//               break;
//            }
//            lr *= 0.95;
//         }

         CC model = new CC(this);
         model.weights = cwt.getWeights();
         return model;
      }
   }

   public static class CC extends Classifier {
      WeightMatrix weights;

      protected CC(ClassifierLearner learner) {
         super(learner);
      }


      @Override
      public Classification classify(Vector v) {
         return createResult(weights.dot(v, Activation.SOFTMAX).toArray());
      }
   }

   public static class LrClassifier extends BinaryGLM {

      /**
       * Instantiates a new Classifier.
       *
       * @param learner the learner
       */
      LrClassifier(ClassifierLearner learner) {
         super(learner);
      }

      @Override
      public Classification classify(Vector vector) {
         double[] dist = new double[2];
         dist[1] = 1.0 / (1.0 + Math.exp(-(weights.dot(vector) + bias)));
         dist[0] = 1.0 - dist[1];
         return createResult(dist);
      }
   }
}// END OF LrLearner
