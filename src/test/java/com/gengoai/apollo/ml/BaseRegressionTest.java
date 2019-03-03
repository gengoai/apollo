package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.data.DatasetType;
import com.gengoai.apollo.ml.data.format.CSVDataFormat;
import com.gengoai.apollo.ml.data.format.DataFormat;
import com.gengoai.apollo.ml.params.ParamValuePair;
import com.gengoai.apollo.ml.regression.Regression;
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
   private final Regression regression;
   private final ParamValuePair[] fitParameters;


   public BaseRegressionTest(Regression regression,
                             ParamValuePair... fitParameters
                            ) {
      this.regression = regression;
      this.fitParameters = fitParameters;
   }

   @Test
   public void fitAndEvaluate() {
      assertTrue(passes(RegressionEvaluation.crossValidation(airfoilDataset(),
                                                             regression,
                                                             3, false, fitParameters)));
   }


   public abstract boolean passes(RegressionEvaluation mce);

   public Dataset airfoilDataset() {
      DataFormat csv = new CSVDataFormat(CSV.builder()
                                            .delimiter('\t')
                                            .header("Frequency", "Angle", "Chord", "Velocity", "Suction", "Pressure"),
                                         "Pressure");
      try {
         return DatasetType.InMemory.read(
            Resources.fromClasspath("com/gengoai/apollo/ml/airfoil_self_noise.data"), csv);
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }
}//END OF BaseRegressionTest
