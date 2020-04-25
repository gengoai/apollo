package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.evaluation.SilhouetteEvaluation;
import com.gengoai.apollo.ml.model.clustering.KMeans;

/**
 * @author David B. Bracewell
 */
public class KMeansTest extends BaseClustererTest {

   public KMeansTest() {
      super(new KMeans(p -> p.K.set(10)));
   }

   @Override
   public boolean passes(SilhouetteEvaluation mce) {
      return mce.getAvgSilhouette() >= 0.9;
   }

}//END OF KMeansTest
