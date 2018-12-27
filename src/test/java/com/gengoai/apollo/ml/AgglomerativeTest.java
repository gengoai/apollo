package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.AgglomerativeClusterer;
import com.gengoai.apollo.ml.clustering.Clustering;
import com.gengoai.apollo.ml.clustering.HierarchicalClustering;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.apollo.ml.preprocess.FilterPreprocessor;
import com.gengoai.conversion.Cast;

/**
 * @author David B. Bracewell
 */
public class AgglomerativeTest extends BaseClustererTest {

   public AgglomerativeTest() {
      super(new AgglomerativeClusterer(new FilterPreprocessor("AutoColumn_0")),
            new AgglomerativeClusterer.Parameters());
   }


   @Override
   public Clustering convertClustering(Clustering clustering) {
      return Cast.<HierarchicalClustering>as(clustering).asFlat(4000);
   }

   @Override
   public boolean passes(SilhouetteEvaluation mce) {
      return mce.getAvgSilhouette() >= 0.85;
   }


}//END OF KMeansTest
