package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.classification.ClassifierEvaluation;
import com.gengoai.apollo.ml.classification.LibLinearModel;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.conversion.Cast;

/**
 * @author David B. Bracewell
 */
public class LibLinearClassifierTest extends BaseClassifierTest {

   public LibLinearClassifierTest() {
      super(new LibLinearModel(), Model.verbose.set(false));
   }

   @Override
   public boolean passes(ClassifierEvaluation evaluation) {
      MultiClassEvaluation mce = Cast.as(evaluation);
      return mce.microF1() >= 0.85;
   }
}//END OF LibLinearClassifierTest
