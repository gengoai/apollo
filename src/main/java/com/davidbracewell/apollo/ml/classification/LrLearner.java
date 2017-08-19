package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.nn.slt.FeedForwardNetworkLearner;
import com.davidbracewell.apollo.ml.classification.nn.slt.OutputLayer;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.apollo.optimization.*;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import com.davidbracewell.apollo.optimization.update.DeltaRule;
import com.davidbracewell.guava.common.base.Stopwatch;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.tuple.Tuple2;
import lombok.val;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.util.Random;

import static com.davidbracewell.apollo.ml.classification.ClassifierEvaluation.crossValidation;
import static com.davidbracewell.tuple.Tuples.$;

/**
 * Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
 *
 * @author David B. Bracewell
 */
public class LrLearner extends BinaryClassifierLearner {

   public static FloatMatrix dsigmoid(FloatMatrix in) {
      return in.mul(FloatMatrix.ones(in.rows, in.columns).subi(in));
   }

   public static void main(String[] args) {
      Resource url = Resources.from(
         "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/iris_headers.csv");


      DenseCSVDataSource dataSource = new DenseCSVDataSource(url, true);
      dataSource.setLabelName("Species");
      Dataset<Instance> dataset = Dataset.classification()
                                         .source(dataSource)
                                         .shuffle(new Random(1234));

      val timer = Stopwatch.createStarted();
      crossValidation(dataset, () -> {
//         return new SoftmaxLearner();
         return FeedForwardNetworkLearner.builder()
                                         .maxIterations(300)
                                         .batchSize(32)
                                         .reportInterval(-1)
                                         .tolerance(1e-9)
                                         .learningRate(new ConstantLearningRate(0.1))
                                         .layer(OutputLayer.softmax())
                                         .build();
      }, 10)
         .output(System.out);
      System.out.println(timer);


//      dataset.encode();
//      int nL = dataset.getLabelEncoder().size();
//      int nF = dataset.getFeatureEncoder().size();
//      int X_size = dataset.size();
//      FloatMatrix X = FloatMatrix.zeros(nF, X_size);
//      FloatMatrix Y = FloatMatrix.zeros(nL, X_size);
//      dataset.asVectors()
//             .zipWithIndex()
//             .forEach((vec, column) -> {
//                         Y.put((int) vec.getLabelAsDouble(), column.intValue(), 1.0f);
//                         X.putColumn(column.intValue(),
//                                     new FloatMatrix(vec.dimension(), 1,
//                                                     Convert.convert(vec.toArray(),
//                                                                     float[].class)));
//                      }
//                     );
//      X.subiColumnVector(X.rowMeans())
//       .diviColumnVector(X.rowSums());
//
//      Stopwatch sw = Stopwatch.createStarted();
//
//      FloatMatrix w1 = rand(50, nF);
//      FloatMatrix w2 = rand(nL, 50);
//      FloatMatrix b1 = FloatMatrix.zeros(50, 1);
//      FloatMatrix b2 = FloatMatrix.zeros(nL, 1);
//      float lr = 0.005f;
//      for (int i = 0; i < 3000; i++) {
//         //Forward prop
//         val z1 = (w1.mmul(X)).addiColumnVector(b1);
//         val a1 = sigmoid(z1);
//         val z2 = (w2.mmul(a1)).addiColumnVector(b2);
//         val a2 = softmax(z2);
//
//         double cost = log(a2).mul(Y)
//                              .add(log(a2.rsub(1.0f)).mul(Y.rsub(1.0f)))
//                              .sum();
//
//         System.out.println(cost);
//
//         //backward prop
//         val dz2 = a2.sub(Y);
//         val dw2 = dz2.mmul(a1.transpose()).divi(X_size);
//         val db2 = dz2.rowSums().divi(X_size);
//         val dz1 = w2.transpose().mmul(dz2).muli(dsigmoid(a1));
//         val dw1 = dz1.mmul(X.transpose()).divi(X_size);
//         val db1 = dz1.rowSums().divi(X_size);
//
//         //Weight update
//         w2.subi(dw2.muli(lr));
//         b2.subi(db2.muli(lr));
//         w1.subi(dw1.muli(lr));
//         b1.subi(db1.muli(lr));
//      }
//
//      System.out.println(sw);
//      val a1 = sigmoid((w1.mmul(X)).addiColumnVector(b1));
//      val a2 = softmax((w2.mmul(a1)).addiColumnVector(b2));
//      val pred_Y = a2.columnArgmaxs();
//      val gold_Y = Y.columnArgmaxs();
//      double correct = 0;
//      for (int i = 0; i < pred_Y.length; i++) {
//         if (pred_Y[i] == gold_Y[i]) {
//            correct++;
//         }
//      }
//      System.out.println((correct / pred_Y.length));
//      if (Math.random() < 2) {
//         System.exit(0);
//      }
//
//
//      crossValidation(dataset, () ->
////      new SoftmaxLearner(),
//                                  FeedForwardNetworkLearner.builder()
//                                                           .layer(DenseLayer.relu().outputSize(50))
//                                                           .layer(OutputLayer.softmax())
//                                                           .learningRate(new ConstantLearningRate(0.1))
//                                                           .maxIterations(300)
//                                                           .batchSize(20)
//                                                           .reportInterval(0)
//                                                           .build(),
//                      10).output(System.out);
//      sw.stop();
//      System.out.println(sw);

   }

   public static FloatMatrix rand(int numRows, int numCols) {
      float max = (float) (Math.sqrt(6.0) / Math.sqrt(numCols + numRows));
      float min = -max;
      FloatMatrix f = FloatMatrix.zeros(numRows, numCols);
      for (int i = 0; i < f.length; i++) {
         f.data[i] = (float) (min + (max - min) * Math.random());
      }
      return f;
   }

   public static Tuple2<Integer, Integer> shape(FloatMatrix m) {
      return $(m.rows, m.columns);
   }

   public static FloatMatrix sigmoid(FloatMatrix in) {
      return in.gt(0);
//      return MatrixFunctions.expi(in.neg())
//                            .addi(1.0f).rdiv(1.0f);
   }

   public static FloatMatrix softmax(FloatMatrix in) {
      val max = in.columnMaxs();
      val exp = MatrixFunctions.exp(in.subRowVector(max));
      val sums = exp.columnSums();
      return exp.diviRowVector(sums);
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
