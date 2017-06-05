package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;

import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class PerLearner extends BinaryClassifierLearner {

   public static void main(String[] args) {
      Resource url = Resources.from(
         "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/iris_binned.csv");
//    //"http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale");
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
//      ));
      ClassifierEvaluation eval = ClassifierEvaluation.crossValidation(dataset,
                                                                       () -> new PerLearner().oneVsRest(),
                                                                       10);
      eval.output(System.out);
   }

   @Override
   public void reset() {

   }

   @Override
   protected Classifier trainForLabel(Dataset<Instance> dataset, double trueLabel) {
      //Minibatch Logistic Regression no regularization
      Vector weights = new DenseVector(dataset.getEncoderPair().numberOfFeatures());
      AtomicDouble bias = new AtomicDouble();
      EncoderPair encoder = dataset.getEncoderPair();
      AtomicDouble learningRate = new AtomicDouble(1);

      for (int epoch = 0; epoch < 1_000; epoch++) {
         dataset.shuffle().stream().partition(50).forEach(batch -> {
            double gradient = 0;
            Vector gradW = new SparseVector(encoder.numberOfFeatures());
            AtomicDouble gradB = new AtomicDouble();
            for (Instance instance : batch) {
               double c = instance.stream().mapToDouble(
                  f -> f.getValue() * weights.get((int) encoder.encodeFeature(f.getName()))).sum() + bias.get();
               double s = 1 / (1.0 + Math.exp(-c));
               gradient += s - (encoder.encodeLabel(instance.getLabel()) == trueLabel ? 1 : 0);
               for (Feature feature : instance) {
                  gradW.increment((int) encoder.encodeFeature(feature.getName()), gradient * feature.getValue());
               }
               gradB.addAndGet(gradient);
            }
            for (Vector.Entry de : Collect.asIterable(gradW.nonZeroIterator())) {
               weights.decrement(de.index, learningRate.get() * de.getValue() / 50);
            }
            bias.set(bias.get() * gradB.get());
         });
//         learningRate.set(learningRate.get() * 0.95);
      }
      BinaryGLM glm = new BinaryGLM(dataset.getEncoderPair(),
                                    dataset.getPreprocessors()
      );
      glm.bias = bias.get();
      glm.weights = weights;
      return glm;
   }


}// END OF PerLearner
