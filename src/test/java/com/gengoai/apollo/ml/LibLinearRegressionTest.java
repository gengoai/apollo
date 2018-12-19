package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.regression.LibLinearLinearRegression;
import com.gengoai.apollo.ml.regression.PipelinedRegression;
import com.gengoai.apollo.ml.regression.RegressionEvaluation;

/**
 * @author David B. Bracewell
 */
public class LibLinearRegressionTest extends BaseRegressionTest {

   public LibLinearRegressionTest() {
      super(new PipelinedRegression(new LibLinearLinearRegression(), new PreprocessorList()),
            new LibLinearLinearRegression.Parameters());
   }

   @Override
   public boolean passes(RegressionEvaluation mce) {
      return mce.r2() >= 0.9;
   }
}//END OF LibLinearRegressionTest
