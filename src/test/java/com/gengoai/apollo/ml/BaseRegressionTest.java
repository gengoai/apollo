package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.data.CSVDataSource;
import com.gengoai.apollo.ml.data.DataSource;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.data.DatasetType;
import com.gengoai.apollo.ml.regression.PipelinedRegression;
import com.gengoai.apollo.ml.regression.RegressionEvaluation;
import com.gengoai.io.CSV;
import com.gengoai.io.Resources;
import org.junit.Test;

import java.io.IOException;

import static junit.framework.TestCase.assertTrue;

/**
 * @author David B. Bracewell
 */
public abstract class BaseRegressionTest {
   private final PipelinedRegression classifier;
   private final FitParameters fitParameters;


   public BaseRegressionTest(PipelinedRegression classifier,
                             FitParameters fitParameters
                            ) {
      this.classifier = classifier;
      this.fitParameters = fitParameters;
   }

   @Test
   public void fitAndEvaluate() {
      assertTrue(passes(RegressionEvaluation.crossValidation(airfoilDataset(),
                                                             classifier,
                                                             fitParameters,
                                                             10)));
   }


   public abstract boolean passes(RegressionEvaluation mce);

   public Dataset airfoilDataset() {
      DataSource csv = new CSVDataSource(CSV.builder()
                                            .delimiter('\t')
                                            .header("Frequency", "Angle", "Chord", "Velocity", "Suction", "Pressure"), "Pressure");
      try {
         return DatasetType.InMemory.load(Resources.fromClasspath("com/gengoai/apollo/ml/airfoil_self_noise.data"), csv);
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }
}//END OF BaseRegressionTest
