package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.AgglomerativeClusterer;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.ml.clustering.HierarchicalClusterer;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.apollo.ml.preprocess.FilterPreprocessor;
import com.gengoai.conversion.Cast;

/**
 * @author David B. Bracewell
 */
public class AgglomerativeTest extends BaseClustererTest {

   public AgglomerativeTest() {
      super(new AgglomerativeClusterer(new FilterPreprocessor("AutoColumn_0")));
   }


   @Override
   public Clusterer convertClustering(Clusterer clustering) {
      return Cast.<HierarchicalClusterer>as(clustering).asFlat(4000);
   }

   @Override
   public boolean passes(SilhouetteEvaluation mce) {
      return mce.getAvgSilhouette() >= 0.85;
   }


}//END OF KMeansTest
