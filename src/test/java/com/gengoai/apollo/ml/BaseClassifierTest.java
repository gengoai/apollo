package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.apollo.ml.classification.PipelinedClassifier;
import com.gengoai.apollo.ml.data.CSVDataSource;
import com.gengoai.apollo.ml.data.DataSource;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.data.DatasetType;
import com.gengoai.io.CSV;
import com.gengoai.io.Resources;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class BaseClassifierTest {
   private final PipelinedClassifier classifier;
   private final FitParameters fitParameters;


   public BaseClassifierTest(PipelinedClassifier classifier,
                             FitParameters fitParameters
                            ) {
      this.classifier = classifier;
      this.fitParameters = fitParameters;
   }

   @Test
   public void fitAndEvaluate() {
      assertTrue(passes(MultiClassEvaluation.crossValidation(irisDataset(),
                                                             classifier,
                                                             fitParameters,
                                                             10)));
   }


   public abstract boolean passes(MultiClassEvaluation mce);

   public Dataset irisDataset() {
      DataSource csv = new CSVDataSource(CSV.builder().hasHeader(true), "class");
      try {
         return DatasetType.InMemory.load(Resources.fromClasspath("com/gengoai/apollo/ml/iris.csv"), csv);
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }

}//END OF BaseClassifierTest
