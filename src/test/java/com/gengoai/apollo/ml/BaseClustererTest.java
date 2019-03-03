package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.data.DatasetType;
import com.gengoai.apollo.ml.data.format.CSVDataFormat;
import com.gengoai.apollo.ml.data.format.DataFormat;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.ml.params.ParamValuePair;
import com.gengoai.io.CSV;
import com.gengoai.io.Resources;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class BaseClustererTest {
   private final Clusterer clusterer;
   private final ParamValuePair[] fitParameters;

   public BaseClustererTest(Clusterer clusterer, ParamValuePair... fitParameters) {
      this.clusterer = clusterer;
      this.fitParameters = fitParameters;
   }


   public Clusterer convertClustering(Clusterer clustering) {
      return clustering;
   }

   @Test
   public void fitAndEvaluate() {
      ParamMap pm = clusterer.getDefaultFitParameters();
      pm.update(fitParameters);
      SilhouetteEvaluation evaluation = new SilhouetteEvaluation(pm.get(Clusterer.clusterMeasure));
      clusterer.fit(loadWaterData(), fitParameters);
      evaluation.evaluate(convertClustering(clusterer));
      assertTrue(passes(evaluation));
   }

   protected Dataset loadWaterData() {
      DataFormat csv = new CSVDataFormat(CSV.builder(), -1);
      try {
         return DatasetType.InMemory.read(Resources.fromClasspath("com/gengoai/apollo/ml/water-treatment.data"),
                                          csv);
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }


   protected abstract boolean passes(SilhouetteEvaluation evaluation);

}//END OF BaseClustererTest
