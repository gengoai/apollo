package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.KMeans;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;

/**
 * @author David B. Bracewell
 */
public class KMeansTest extends BaseClustererTest {

   public KMeansTest() {
      super(new KMeans(),
            new KMeans.Parameters().set("K", 10));
   }

   @Override
   public boolean passes(SilhouetteEvaluation mce) {
      return mce.getAvgSilhouette() >= 0.9;
   }


}//END OF KMeansTest
