package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.KMeans;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.conversion.Cast;

/**
 * @author David B. Bracewell
 */
public class KMeansTest extends BaseClustererTest<KMeans.Parameters> {

   public KMeansTest() {
      super(new KMeans(),
            Cast.as(new KMeans.Parameters().set("K", 10)));
   }

   @Override
   public boolean passes(SilhouetteEvaluation mce) {
      return mce.getAvgSilhouette() >= 0.9;
   }


}//END OF KMeansTest
