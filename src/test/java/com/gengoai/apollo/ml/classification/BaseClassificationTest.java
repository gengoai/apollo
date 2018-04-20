package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.data.source.DenseCSVDataSource;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.preprocess.transform.RescaleTransform;
import com.gengoai.io.Resources;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class BaseClassificationTest {

   final ClassifierLearner learner;
   final double expectedAccuracy;
   final double delta;


   public BaseClassificationTest(ClassifierLearner learner, double expectedAccuracy, double delta) {
      this.learner = learner;
      this.expectedAccuracy = expectedAccuracy;
      this.delta = delta;
   }


   public Dataset<Instance> getDataset() {
      DenseCSVDataSource dataSource = new DenseCSVDataSource(
         Resources.fromClasspath("com/gengoai/apollo/ml/iris.csv"),
         true);
      dataSource.setLabelName("class");
      return Dataset.classification().source(dataSource)
                    .preprocess(PreprocessorList.create(new RescaleTransform(0, 1, true)));

   }

   @Test
   public void test() {
      Classifier clf = learner.train(getDataset());
      ClassifierEvaluation evaluation = ClassifierEvaluation.evaluateModel(clf, getDataset());
      System.out.printf("ClassificationTest [%s] accuracy=%f, acceptable_accuracy=%f [- %f]\n",
                        learner.getClass().getSimpleName(), evaluation.accuracy(), expectedAccuracy, delta);
      assertTrue(evaluation.accuracy() >= (expectedAccuracy - delta));
   }


}//END OF BaseClassificationTest
