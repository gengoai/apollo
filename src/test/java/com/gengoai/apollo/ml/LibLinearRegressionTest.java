package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.preprocess.PerFeatureTransform;
import com.gengoai.apollo.ml.preprocess.RescaleTransform;
import com.gengoai.apollo.ml.regression.LibLinearLinearRegression;
import com.gengoai.apollo.ml.regression.RegressionEvaluation;

/**
 * @author David B. Bracewell
 */
public class LibLinearRegressionTest extends BaseRegressionTest {

   public LibLinearRegressionTest() {
      super(new LibLinearLinearRegression(new PerFeatureTransform(RescaleTransform::new)),
            new LibLinearLinearRegression.Parameters()
               .set("maxIterations", 200)
               .set("eps", 1e-10)
               .set("bias", true));
   }

   @Override
   public boolean passes(RegressionEvaluation mce) {
      return mce.rootMeanSquaredError() <= 125;
   }
}//END OF LibLinearRegressionTest
