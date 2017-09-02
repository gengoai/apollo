package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.nn.FeedForwardNetworkLearner;
import com.davidbracewell.apollo.ml.classification.nn.OutputLayer;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.guava.common.base.Stopwatch;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import lombok.val;

import java.util.Random;

import static com.davidbracewell.apollo.ml.classification.ClassifierEvaluation.crossValidation;

/**
 * Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
 *
 * @author David B. Bracewell
 */
public class LrLearner {


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
                                         .batchSize(1)
                                         .reportInterval(-1)
                                         .tolerance(1e-9)
//                                         .learningRate(new ConstantLearningRate(0.1))
                                         .layer(OutputLayer.softmax())
                                         .build();
      }, 10)
         .output(System.out);
      System.out.println(timer);

   }
}// END OF LrLearner
