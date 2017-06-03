package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.TrainTestSplit;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.optimization.BatchOptimizer;
import com.davidbracewell.apollo.optimization.DecayLearningRate;
import com.davidbracewell.apollo.optimization.Optimizer;
import com.davidbracewell.apollo.optimization.SGD;
import com.davidbracewell.apollo.optimization.activation.SoftmaxFunction;
import com.davidbracewell.apollo.optimization.regularization.NonRegularizedDeltaRule;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import lombok.NonNull;

import java.util.Random;

/**
 * Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
 *
 * @author David B. Bracewell
 */
public class LrLearner extends BinaryClassifierLearner {

   public static void main(String[] args) {
      Resource url = Resources.from(
         "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/iris_headers.csv");
//    "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale");
      DenseCSVDataSource dataSource = new DenseCSVDataSource(url, true);
      dataSource.setLabelName("Species");
      Dataset<Instance> dataset = Dataset.classification()
                                         .source(dataSource)
                                         .shuffle(new Random(1234));
//      dataset.preprocess(PreprocessorList.create(
//         new ZScoreTransform("Sepal length"),
//         new ZScoreTransform("Sepal width"),
//         new ZScoreTransform("Petal length"),
//         new ZScoreTransform("Petal width")
//                                                ));

      dataset.encode();
      Optimizer sgd = new BatchOptimizer(new SGD(), 20);

      TrainTestSplit<Instance> tt = dataset.shuffle().split(0.8).iterator().next();
//      tt.getTrain().preprocess(PreprocessorList.create(new RescaleTransform(0, 1, true)));
//      tt.getTrain().preprocess(PreprocessorList.create(
//         new ZScoreTransform("Sepal length"),
//         new ZScoreTransform("Sepal width"),
//         new ZScoreTransform("Petal length"),
//         new ZScoreTransform("Petal width")
//                                                      ));


//      LearningRate learningRate = new ConstantLearningRate(0.1);
//      //new DecayLearningRate(0.1, 0.01);
//      WeightUpdater updater = new L1Regularization(0.01);
//
//      Weights weights = sgd.optimize(
//         Weights.multiClass(dataset.getLabelEncoder().size(), dataset.getFeatureEncoder().size()),
//         () -> tt.getTrain().asFeatureVectors(),
//         new LogisticCostFunction(),
//         TerminationCriteria.create().maxIterations(100).historySize(3).tolerance(1e-3),
//         learningRate,
//         updater,
//         true).getWeights();
//
//
//      SoftmaxFunction softmaxFunction = new SoftmaxFunction();
//      double[] correct = new double[dataset.getLabelEncoder().size()];
//      double[] total = new double[dataset.getLabelEncoder().size()];
//      tt.getTest().forEach(i -> {
//         FeatureVector v = tt.getTrain().getPreprocessors().apply(i).toVector(tt.getTrain().getEncoderPair());
//         Vector p = softmaxFunction.apply(weights.dot(v));
//         double max = p.maxIndex();
//         total[v.getLabel().intValue()]++;
//         if (max == v.getLabel()) {
//            correct[v.getLabel().intValue()]++;
//         }
//      });
//
//      double cSum = 0;
//      double tSum = 0;
//      for (int i = 0; i < correct.length; i++) {
//         System.out.println(correct[i] + ", " + total[i] + " : " + (correct[i] / total[i]));
//         cSum += correct[i];
//         tSum += total[i];
//      }
//      System.out.println("\n\n" + (cSum / tSum));

      ClassifierEvaluation eval = ClassifierEvaluation.crossValidation(dataset,
                                                                       () -> {
                                                                          return new SGDLearner()
                                                                                    .setParameter("learningRate",
                                                                                                  new DecayLearningRate(0.1,
                                                                                                                        0.001))
                                                                                    .setParameter("weightUpdater",
                                                                                                  new NonRegularizedDeltaRule())
                                                                                    .setParameter("activation",
                                                                                                  new SoftmaxFunction())
                                                                                    .setParameter("batchSize", 0);
                                                                       },
                                                                       10
                                                                      );
      eval.output(System.out);
   }

   @Override
   public void reset() {

   }

   @Override
   protected LrClassifier trainForLabel(Dataset<Instance> dataset, double trueLabel) {
      LrClassifier model = new LrClassifier(dataset.getEncoderPair(),
                                            dataset.getPreprocessors());

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

   public static class LrClassifier extends BinaryGLM {

      /**
       * Instantiates a new Classifier.
       *
       * @param encoderPair   the encoder pair
       * @param preprocessors the preprocessors
       */
      protected LrClassifier(EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
         super(encoderPair, preprocessors);
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
